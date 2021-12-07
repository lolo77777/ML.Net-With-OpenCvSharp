using Microsoft.ML;
using Microsoft.ML.Data;

using OpenCvSharp;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sample.OpenCv
{
    public class MLModelHelp
    {
        private static PredictionEngine<ModelInputBytes, ModelOutput> _predEngineBytes;

        #region PrivateFunction

        //更改trainData类型，并调用该方法重新训练
        private static ITransformer RetrainPipelineBytes(MLContext context, IDataView trainData)
        {
            var trainingPipeline = context.Transforms.Conversion.MapValueToKey(@"Label", @"Label")
                .Append(context.MulticlassClassification.Trainers.ImageClassification(labelColumnName: @"Label"))
                .Append(context.Transforms.Conversion.MapKeyToValue(@"PredictedLabel", @"PredictedLabel"));
            var model = trainingPipeline.Fit(trainData);
            return model;
        }

        //将本地的图像转换为byte[]，以进行重新训练
        //重新训练后，在使用mat进行图像预测时，调用mat.ToBytes()作为输入进行预测即可
        private static byte[] GetMatBytes(string imagePath)
        {
            var mat = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
            var vs = mat.ToBytes();
            return vs;
        }

        private static ModelOutput PredictFromBytes(ModelInputBytes input)
        {
            ModelOutput result = _predEngineBytes.Predict(input);
            return result;
        }

        #endregion PrivateFunction

        #region PublicFunction

        /// <summary>
        /// 重新训练
        /// </summary>
        public static void ReTrain()
        {
            //读取原来的样本图像，转换为新的模型输入类型集合
            var input1 = Directory.GetFiles("./Data/Image/Front/").Select(f => new ModelInputBytes()
            {
                Label = "Front",
                ImageSource = GetMatBytes(f)
            }).ToList();
            var input2 = Directory.GetFiles("./Data/Image/Reverse").Select(f => new ModelInputBytes()
            {
                Label = "Reverse",
                ImageSource = GetMatBytes(f)
            }).ToList();
            input1.AddRange(input2);

            MLContext mlContext = new MLContext();
            //从集合加载，生成新的数据样本
            IDataView newData = mlContext.Data.LoadFromEnumerable<ModelInputBytes>(input1);

            // Retrain model
            var retrainedModel = RetrainPipelineBytes(mlContext, newData);
            mlContext.Model.Save(retrainedModel, newData.Schema, "newModel.zip");
        }

        public static string Predict(Mat mat)
        {
            var bytes = mat.ToBytes();
            var output = PredictFromBytes(new ModelInputBytes { ImageSource = bytes });
            return output.Prediction;
        }

        /// <summary>
        /// 初始化模型
        /// </summary>
        /// <param name="MLNetModelPathBytes"></param>
        public static void CreatePredictEngineBytes(string MLNetModelPathBytes)
        {
            MLContext mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPathBytes, out var modelInputSchema);
            _predEngineBytes = mlContext.Model.CreatePredictionEngine<ModelInputBytes, ModelOutput>(mlModel);
        }

        #endregion PublicFunction
    }

    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }

        public float[] Score { get; set; }
    }

    public class ModelInputBytes
    {
        [ColumnName(@"Label")]
        public string Label { get; set; }

        [ColumnName(@"Features")]
        public byte[] ImageSource { get; set; }
    }
}
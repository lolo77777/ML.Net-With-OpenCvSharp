using OpenCvSharp;

using System;
using System.IO;
using System.Linq;

namespace Sample.OpenCv
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            ////重新训练，保存新的模型 newModel.zip
            //MLModelHelp.ReTrain();

            ////加载新模型，请确认完成上述训练，存在模型
            MLModelHelp.CreatePredictEngineBytes("./newModel.zip");
            //使用测试的图像进行测试
            var fold1 = "./Data/Test/Front";
            var fold2 = "./Data/Test/Reverse";
            var mats = Directory.GetFiles(fold1).Select(f => Cv2.ImRead(f)).ToList();
            var mats2 = Directory.GetFiles(fold2).Select(f => Cv2.ImRead(f)).ToList();
            foreach (Mat mat in mats)
            {
                Console.WriteLine(MLModelHelp.Predict(mat));
            }
            foreach (Mat mat in mats2)
            {
                Console.WriteLine(MLModelHelp.Predict(mat));
            }
            Console.WriteLine("Complete!");
            Console.Read();
        }
    }
}
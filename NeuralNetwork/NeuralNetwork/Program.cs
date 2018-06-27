using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {

            // 000 -> 0
            // 001 -> 1
            // 010 -> 1
            // 011 -> 0
            // 100 -> 1
            // 101 -> 0
            // 110 -> 0
            // 111 -> 1

            NeuralNetwork neuralNetwork = new NeuralNetwork(new int[] { 3, 40, 40, 40, 40, 1 });

            for (int help = 0; help < 8000; help++)
            {

                neuralNetwork.FeedForward(new float[] { 0, 0, 0 });
                neuralNetwork.BackwardsPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 0, 0, 1 });
                neuralNetwork.BackwardsPropagation(new float[] { 1 });

                neuralNetwork.FeedForward(new float[] { 0, 1, 0 });
                neuralNetwork.BackwardsPropagation(new float[] { 1 });

                neuralNetwork.FeedForward(new float[] { 0, 1, 1 });
                neuralNetwork.BackwardsPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 1, 0, 0 });
                neuralNetwork.BackwardsPropagation(new float[] { 1 });

                neuralNetwork.FeedForward(new float[] { 1, 0, 1 });
                neuralNetwork.BackwardsPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 1, 1, 0 });
                neuralNetwork.BackwardsPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 1, 1, 1 });
                neuralNetwork.BackwardsPropagation(new float[] { 1 });


            }

            Console.WriteLine( neuralNetwork.FeedForward(new float[] { 0, 0, 0 })[0]);
            Console.WriteLine( neuralNetwork.FeedForward(new float[] { 0, 0, 1 })[0]);
            Console.WriteLine( neuralNetwork.FeedForward(new float[] { 0, 1, 0 })[0]);
            Console.WriteLine( neuralNetwork.FeedForward(new float[] { 0, 1, 1 })[0]);
            Console.WriteLine( neuralNetwork.FeedForward(new float[] { 1, 0, 0 })[0]);
            Console.WriteLine( neuralNetwork.FeedForward(new float[] { 1, 0, 1 })[0]);
            Console.WriteLine( neuralNetwork.FeedForward(new float[] { 1, 1, 0 })[0]);
            Console.WriteLine( neuralNetwork.FeedForward(new float[] { 1, 1, 1 })[0]);

            Console.ReadLine();

        }
    }
}

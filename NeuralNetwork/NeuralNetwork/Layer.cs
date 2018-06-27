using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    /// <summary>
    /// Warstwa sieci neuronowej
    /// </summary>
    class Layer
    {

        /// <summary>
        /// liczba neuronów wejściowych w warstwie
        /// </summary>
        public int quantityOfInputs;

        /// <summary>
        /// liczba neurów wyjściowych w warstwie
        /// </summary>
        public int quantityOfOutputs;

        /// <summary>
        /// Wejścia warstwy
        /// </summary>
        public float[] inputs;

        /// <summary>
        /// Wyjścia Warstwy
        /// </summary>
        public float[] outputs;

        /// <summary>
        /// Wagi
        /// </summary>
        public float[,] weights;

        /// <summary>
        /// wagi po aktualizacji - zmiana wag oryginalnych przy pomocy algorytmu propagacji wstecznej
        /// </summary>
        public float[,] weightsAfterUpDate;

        /// <summary>
        /// ( O(n+1) - y(n) ) * ( 1 - O(n+1)^2) - gamma, aby nie liczyć tego za każdym razem 
        /// </summary>
        public float[] gamma;

        /// <summary>
        /// błąd
        /// </summary>
        public float[] error;
        
        /// <summary>
        /// konstruktor parametryczny, inicjalizuje wszystkie pola
        /// </summary>
        /// <param name="quantityOfInputs"> liczba wejść do warstwy sieci neuronowej </param>
        /// <param name="quantityOfOutputs"> liczba wyjść z warstwy sieci neuronowej </param>
        public Layer( int quantityOfInputs, int quantityOfOutputs )
        {

            this.quantityOfInputs = quantityOfInputs;

            this.quantityOfOutputs = quantityOfOutputs;

            this.inputs = new float[ quantityOfInputs ];

            this.outputs = new float[ quantityOfOutputs ];

            this.weights = new float[ quantityOfOutputs, quantityOfInputs ];

            this.weightsAfterUpDate = new float[ quantityOfOutputs, quantityOfInputs ];

            this.gamma = new float[ quantityOfOutputs ];

            this.error = new float[ quantityOfOutputs ];

            InitializeWeights();

        }

        /// <summary>
        /// Zwraca wyjścia poprzedniej warstwy sieci neuronowej
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public float[] FeedForward( float[] inputs)
        {

            this.inputs = inputs;

            for (int help = 0; help < quantityOfOutputs; help++)
            {

                this.outputs[help] = 0;

                for (int me = 0; me < quantityOfInputs  ; me++)
                {

                    // Sumowanie iloczynów wejść i wag 
                    this.outputs[ help ] += this.inputs[ me ] * weights[ help, me ];

                }

                // tangens hiperboliczny z wyjścia
                this.outputs[help] = (float)Math.Tanh( this.outputs[help] );

            }

            return outputs;

        }


        /// <summary>
        /// Inicjalizacja wag w przedziale ( -0.5, 0.5 )
        /// </summary>
        private void InitializeWeights()
        {

            Random random = new Random();

            for (int help = 0; help < quantityOfOutputs; help++)
            {

                for (int me = 0; me < quantityOfInputs ; me++)
                {

                    weights[help, me] = (float)random.NextDouble() - 0.5f;

                }

            }

        }


        /// <summary>
        /// aktualizacja wag
        /// </summary>
        public void UpDateWeights()
        {

            for (int help = 0; help < quantityOfOutputs; help++)
            {

                for (int me = 0; me < quantityOfInputs; me++)
                {

                    weights[help, me] -= weightsAfterUpDate[help, me] * 0.033f ; 

                }   

            }

        }

        /// <summary>
        /// Propagacja wsteczna dla wyjścia
        /// </summary>
        /// <param name="expected"></param>
        public void BackwardPropagationOutput( float[] expected )
        {

            for (int help = 0; help < quantityOfOutputs; help++)
            {

                error[help] = outputs[help] - expected[help];

                gamma[help] = error[help] * HyperbolicTangensDerrivative(outputs[help]);

                for (int me = 0; me < quantityOfInputs; me++)
                {

                    weightsAfterUpDate[help, me] = gamma[help] * inputs[me];

                }


            }

        }
        

        /// <summary>
        /// 
        /// </summary>
        /// <param name="gammaForward"></param>
        /// <param name="forward"></param>
        public void BackwardPropagationHidden(float[] gammaForward, float[,] weightsForward ) {

            for (int help = 0; help < quantityOfOutputs; help++)
            {

                gamma[help] = 0;

                for (int me = 0; me < gammaForward.Length; me++)
                {

                    gamma[help] += gammaForward[me] * weightsForward[me,help];

                }

                gamma[ help ] *= HyperbolicTangensDerrivative( outputs[help] );

            }

            for (int help = 0; help < quantityOfOutputs; help++)
            {

                for (int me = 0; me < quantityOfInputs; me++)
                {

                    weightsAfterUpDate[help, me] = gamma[help] * inputs[me];

                }

            }

        }

        /// <summary>
        /// Pochodna z tangensa hiperbolicznego
        /// </summary>
        /// <param name="value"> argumetn funkcji </param>
        /// <returns></returns>
        public float HyperbolicTangensDerrivative( float value )
        {

            return 1 - (value * value);

        }

    }

}


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    /// <summary>
    /// Sieć neuronowa
    /// </summary>
    class NeuralNetwork
    {

        /// <summary>
        /// Liczba warstw sieci neuronowych
        /// </summary>
        int[] layer;

        /// <summary>
        /// Warstwy sieci neuronowej
        /// </summary>
        Layer[] layers;

        /// <summary>
        /// Konstruktor parametryczny
        /// </summary>
        /// <param name="layer"> int[] - liczba warstw sieci neuronowej </param>
        public NeuralNetwork( int[] layer  )
        {

            this.layer = new int[layer.Length];

            this.layer = layer;

            this.layers = new Layer[layer.Length - 1];

            for ( int help = 0; help < layers.Length; help++ )
            {

                this.layers[help] = new Layer( layer[ help ], layer[ help + 1 ] );

            }

        }

        /// <summary>
        /// Wartości wyjść ostaniej warstwy sieci neuronowej
        /// </summary>
        /// <param name="inputs"> wejścia </param>
        /// <returns> wyjścia sieci neuronowej </returns>
        public float[] FeedForward( float[] inputs )
        {

            layers[0].FeedForward(inputs);

            for (int help = 1; help < layers.Length; help++)
            {

                layers[help].FeedForward( layers[ help - 1 ].outputs );

            }

            return layers[layers.Length - 1].outputs;

        }

        /// <summary>
        /// Algorytm propagacji wstecznej
        /// </summary>
        /// <param name="expected"></param>
        public void BackwardsPropagation( float[] expected )
        {

            for (int help = layers.Length-1; help >= 0; help--)
            {

                if( help == layers.Length - 1)
                {

                    layers[help].BackwardPropagationOutput(expected);

                }

                else
                {

                    layers[help].BackwardPropagationHidden(layers[help + 1].gamma, layers[help + 1].weights);

                }

            }
            
            foreach( Layer layer in layers)
            {

                layer.UpDateWeights();

            }

        }

    }
}

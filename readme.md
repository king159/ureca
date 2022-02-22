# Apply Different Neural Networks in Evaluation Function of Computer Chess

NTU URECA project (undergraduate research project)

## Abstract and conclusion
This project aims at comparing the performance of different architectures 
of neural network that acts as the **statistic evaluation function** of 
the computer chess engine. We first label 6 million distinct positions of 
games from [ChessBase 16](https://shop.chessbase.com/en/products/chessbase_16_mega_package?ref=RF191-8I2RXB2L67) and labeled by 
[Stockfish 13](https://stockfishchess.org/blog/2021/stockfish-13/). Based on different point of views of the same
position, we train autoencoder classifier with the 1-D vector which represents the position of the pieces, convolutional neural network
with consider the position as a 8x8 image with multiple channels, classifier, and a neural network classifier from hand-craft feature to classify the positions.
Our best neural networks get more than 90% top-2 accuracy in the 
7-class classification task.

## Code structure
1. stockfinsh: Stockfish engines which are used to label the positions into 7 classes 
   1. stockfinsh12.exe
   2. stockfinsh13.exe
2. [autoencoder.ipynb](autoencoder.ipynb): The autoencoder model
3. [CNN.ipynb](CNN.ipynb): The cnn model which is not included in the report due to its unpromising performance
5.[manual.ipynb](manual.ipynb): The model for manual-generated features
4. [pgn2fen.py](pgn2fen.py): The script to convert PGN to FEN
5. [search_best.py](search_best.py): We use [optuna](https://optuna.org/) to find the best parameters
6. [util.py](util.py): Conversion from FEN to different types of tensor for different models
7. [report.pdf](report.pdf): Detailed report of this project
8. scripts to label using the Stockfish engine

   1.[label_asyncio.py](label_asyncio.py)

   2.[label_thread.py](label_thread.py)

9. Due to the commercial constraints and the size of converted tensor file, all training data is not included in this repository.
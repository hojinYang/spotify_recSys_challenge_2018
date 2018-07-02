
# Spotify RecSys Challenge 2018
***hello world!*** Team: Hojin Yang, Minjin Choi, and Yoon Ki Jeong. **Main track.**  
Data Mining Lab, Sungkyunkwan university.   
you can send us an email at hojin.yang7@gamil.com.  

 
**This document is constantly being updated.**


This project is an automatic playlist continuation(APC) system implemented using Tensorflow.  
The main algorithm we use is Denoising Autoencoders which reconstructs original playlists from an input of corrupted ones. 
In additon, we apply Character-level convolutional neural networks to playlist-title based music recommendation.  


The charateristics of our model are as follows: 
* Unlike pure collaborate filtering model which only extends playlists profiled at training time, 
our system generates a list of recommended tracks to a new playlist(not in the training set).

* Not only tracks but also artists are used to construct latent representation of playlists. 
This allows robust recommendation for a playlist of songs occur in very few playlists.

## Development Environment
* Python Anaconda v4.4.10  
* Tensorflow v1.5.0  
* CUDA Toolkit v9.0 and cuDNN v7.0  
* GPU: 4 Nvidia GTX 1080Ti  

## Dataset
Spotify has produced the MPD(Million Playlist Dataset) which contains a million user-curated playlists. 
Each playlist in the MPD contains a playlist title, a list of tracks(with metadata), and other miscellaneous information. 

## Preprocess The Data
Proceed with these steps to convert the MPD’s data format into our system's.
1. Download Spotify's MPD tar file which contains a thousand MPD-slice json files(each contains a thounsand playlists).
2. Divide json files into two groups: training folder and test folder. And place each data folders into the root folder of the project.
3. You can get resturctured data by running **data_generator.py**.  
*Arguments of data_generator.py*   
`--datadir`		: Directory where converted dataset(training, test, challenge) will be stored. *default: ./data*  
`--mpd_tr`		: Directory which contains MPD-slice json files used for training the model. *default: ./mpd_train*  
`--mpd_te`		: Directory which contains MPD-slice json files used for testing the model. *default: ./mpd_test*     
`--mpd_ch`		: Directory which contains the challenge set json file. *default: ./challenge*   
`--mincount_trk`	: The minimum number of occurences of tracks in the train data *default: 5*  
`--mincount_art`	: The minimum number of occurences of artists in the train data *default: 3*  
`--divide_ch`		: A list where each elements is a range of challenge seed numbers *default: 0-1,5,10-25,10-25r* 	    
For example, you can set the dicrectories as the following command : 
```console  
python data_generator.py --datadir ./data --mpd_tr ./mpd_train --mpd_te ./mpd_test --mpd_ch ./challenge
```
You can set the minimum number of occurences of tracks and artists on training set manually. 
When you run the following command, tracks with less than three occurrences are removed:
```console
python data_generator.py --mincount_trk 3 
```
4. Scripts above populate the ‘./data’ with one training json file, multiple types of test json, and challenge json.  

    
Each test files contains same seed pattern as Spotify RecSys Challenge: seed 0, 1, 5, 10, 25, 100, 25r, 100r.  
We also divide challenge set into four categories based on seed pattern by default: (0,1) , (5) , (10,25,100) , (25r,100r)  
 
For submission, we train our models with four different denoising schemes.
Each schemes performs better on one of four different challenge categories.  


## Run The System
Our model is composed of two parts: Denoising Autoencoders and Character-level CNN; 
train the parameters of the DAE first, then integrate with char-level CNN.
1. Create a folder into the root folder of the project. *config.ini* file, which contains information required to run the model, 
must be placed into the created folder(Check the structure of *conf.ini* below).  
2. You can train models by running **main.py**.  
*Arguments of main.py*   
`--dir`			: Directory name which contains config file.  
`--pretrain`	: Pretrain dae parameters if specified.  
`--dae`			: Train dae parameters if specified.  
`--title`		: Train paramters of title module if specified.  
`--challenge`	: Generate challenge submission candidates if specified.  
`--testmode`	: Get the results without training the model if specified.   
Suppose the folder you create at the step above is './sample'.   
we recommand you to **pretrain the DAE with tied condition**; constrain decoder’s weights to be equal to transposed encoder’s.
Using those weights as initial values of DAE brings much better results than not pretraining the weights.   
First, run main in pretrain mode(tied DAE):  
```console
python main.py --dir sample --pretrain
```
Run main in DAE mode after the loss is converged in pretrain mode.
If you set pretrain file name in config.ini file, following command will use pretrained paramters saved in the fold you created(./sample). 
You can also train DAE without initital value depending on the *config.ini* setting:  
```console
python main.py --dir sample --dae
```
After you run DAE, its parameters are saved as pickle format in ./sample.   

3. You can train char-CNN if DAE’s parameters is save in the folder you created. After you run the command below, the final tensor graph will be generated:
```console
python main.py --dir sample --title
```
4. Finally you can generate challenge submission candidates by using graph and DAE paramters you get at the steps above:  
```console
python main.py --dir sample --challenge
```
**[note]**  
For all models, paramters are updated if the avearge of *update_seeds* r-precision score(s) increases. Our system calculates r-precision score every epoch.  
You must specify only one mode(dae, title, challenge) when you set arguments of *main.py*.  
You can easily replace parameter pickle files(for DAE) and/or ckpt graph file(for title) with other directories, 
if both have same number of tracks & artists and same CNN filter shapes.     
If you want to just check metrices scores after replacing paramters with directory's, using *--testmode* is efficient:
```console
# after replacing DAE pickle file from another folder #
python main.py --dir sample --dae --testmode
```

## Build Our Submission
We already set the initial setting: create 4 different directories(0to1_inorder,5_inorder,10to100_inorder, 25to100_random), 
and set config files of each directories.  
1. Divide 1,000 *mpd.slice.#.json files* into two directories(mpd_train, mpd_test). We use 997 slices for training 
except *'mpd.slice.250000-250999', 'mpd.slice.500000-500999', 'mpd.slice.750000-750999'* which are used for testing the model.  
The directory containing *challenge_set.json* is also needed for generating challenge data following our format.  
2. Run **data_generator.py** with default arguments(or change if you want). 
Then './data' is created which contains training data and test data with multiple categories as json format. 
Challenge data with four different categories are also created as we set `--divide_ch` of **data_generator.py** as *0-1,5,10-25,10-25r(andom)*.  
Dividing challenge data into four categories means we use four different denoising schemes to train our model 
and merge the results at the last moment.  
3. We already set four different directories which contain *config.ini* optimized for each challenge categories. 
The approximante information is shown in the table below.  
4. Run in pretrain mode for each directories. Then run in dae mode except 0to1_inorder.  
5. For title mode, it is more efficient to run in just one directory(0to1_inorder) 
and copy the tensor graph outputs(generated after running on title mode) to others. 
You don't have to train in title mode for all directories, as outputs are same.  
6. Run challenge mode for each directories.  
7. Run **merge_results.py** to merge results from different directories and to generate results.csv files.  

| directory | challenge category | firstN_range | input denoising | pretrain only	|
|--------|--------|--------|--------|--------|
| 0to1_inorder | challenge_inorder_0to1 | 0, 0.3 | 0.75 | True |
| 5_inorder | challenge_inorder_5 | 1, 50 | 0.75 | False |
| 10to100_inorder | challenge_inorder_10to100 | 0.3, 0.6 | 0.75 | False |
| 25to100_random | challenge_random_25to100 | -1 | 0.5, 0.8 | False |  

In summary, run the following commands one line at a time:  
```console
# 997 mpd.slice on ./mpd_train, 3 mpd.slice on ./mpd_test, challenge set on ./challenge #  
python data_generator.py  
python main.py --dir 0to1_inorder --pretrain  
python main.py --dir 0to1_inorder --title  
python main.py --dir 0to1_inorder --challenge  
# copy 0to1_inorder/graph to 5_inorder #
python main.py --dir 5_inorder --pretrain  
python main.py --dir 5_inorder --dae  
python main.py --dir 5_inorder --challenge  
# copy 0to1_inorder/graph to 10to100_inorder #
python main.py --dir 10to100_inorder --pretrain  
python main.py --dir 10to100_inorder --dae  
python main.py --dir 10to100_inorder --challenge  
# copy 0to1_inorder/graph to 25to100_inorder #
python main.py --dir 25to100_random --pretrain  
python main.py --dir 25to100_random --dae  
python main.py --dir 25to100_random --challenge  

python merge_results.py
```
**[Note]**  
* It takes about 3\~4 days to train using the whole MPD under our environment.  
* We set every epochs as 20. If the r-precision of *update_seed* continuously decreases and no parameters update occur, 
it is recommended to stop the operation manually and proceed to the next steps. 
Also, more epochs might be needed if you train using small data set.  
* You should modify some lines of code in *models/DAEs.py* if your system has fewer than three GPUs.  

## Sturcture of config.ini
***[BASE]***  
**verbose** - *boolean.* print log on console if True.  
**data_dir** - *string.* Directory of data that system will read. 
The directory contains one training json file and multiple types of test json files.  
**challenge_dir** - *string.* Directory where final results are saved.   
**testsize** - *int.* The number of maximum test playlist in each test case.


***[DAE]***  
**epochs** - *int.* Number of training epoch.  
**batch** - *int.* batch size.  
**lr** - *float.* learning rate.  
**reg_lamdba** - *float.* regularization constant.  
**hidden** - *int.* DAE hidden layer size.  
**test_seed** - *comma seperated int(or int+’r’) list.* Seed numbers that you run the test after each epoch.   
*test_seed =  1,5,10* means the system runs test after each epoch by reading test-1, test-5, test-10 json file in the directory set in fold_dir.  
**update_seed** -  *comma seperated int(or int+’r’) list.* Seed numbers that is considered when updating parameters. Update_seed must be inner set of test_seed.  
*test_seed = 25r,100r , update_seed = 100r* means  the system runs test after each epoch by reading test-25r, test-100r json file, creates log, 
and update parameters if the test-100r’s r-precision  value increases.  
**keep_prob** - *float(0.0<x<=1.0).* Drop out keep probability in hidden layer.  
*keep_prob = 0.75* means drop out 25% of input for every batch.  
**input_kp** - *comma seperated floats list(0.0<x<=1.0).* Denoising keep probability range in input layer.  
*input_kp = 0.5, 0.8* means denoise randomly selected probability between 50%~20%.  
**firstN_range** - *comma seperated floats or int list.* The range to draw a random number n,. 
when you set the tracks from 0th track to n\-th track of a playlist as input value. 
You can set it up in three different ways.  
*firstN_range = -1* means to consider all the songs in the playlist as an input value.  
*firstN_range = float a , float b* means set input track range from 0\-th to random(a\*N, b\*N). (N is the length of the playlist)
*firstN_range = int a , int b* means set input track range from 0\-th to random(a, b).   
ex)  
firstN_range - -1 : 0\~N  
firstN_range - 0,50 : 0\~random(0,50)  
firstN_range - 0.3,0.6 : 0\~random(N\*0.3, N\*0.6)  
**initval** - *string.* Name of pickle file which contains pretrained parameters. Set NULL if no initial value.     
**save** - *string.* Name of pickle file to store the updated parameters.

***[PRETRAIN]***  
**epochs** - *int.* Number of training epoch.  
**batch** - *int.* batch size.  
**lr** - *float.* learning rate.  
**reg_lamdba** - *float.* regularization constant.  
**save** - *string.* Name of pickle file to store the updated parameters.


***[TITLE]***  
**epochs** - *int.* Number of training epoch.  
**batch** - *int.* batch size.  
**lr** - *float.* learning rate.  
**keep_prob** - *float(0.0<x<=1.0).* Drop out keep probability in DAE hidden layer.  
**input_kp** - *comma seperated floats(0.0<x<=1.0).* Denoising keep probability range in  input layer.  
**title_kp** - *float(0.0<x<=1.0).* Drop out keep probability in title model hidden layer.    
**test_seed** - *comma seperated int(or int+’r’) list.* Seed numbers that you run the test after each epoch.   
**update_seed** -  *comma seperated int(or int+’r’) list.* Seed numbers that is considered when updating parameters.   
**char_model** - *Char_CNN or Char_RNN*  
**rnn_hidden** - *int.* Set this one if char_model is Char_RNN. RNN hidden size.  
**filter_num** - *int.* Set this one if char_model is Char_CNN. Number of CNN filters.  
**filter_size** - *comma seperated int list.* Set this one if char_model is Char_CNN. Size of CNN filters.  
**char_emb** - *int.* Character embedding size. One-hot if the value is 0.  
**DAEval** - *string.* Name of pickle file where the parameters of DAE is saved.  
**save** - *string.* Name of checkpoint file which saves updated tensor graph.



***[CHALLENGE]***  
**batch** - *int.* batch size.  
**challenge_data** - *string.* Name of challenge file whose format is modified to fit our system in *data-dir*.  
**result** - *string.* Name of pifckle file to save the result.  
  
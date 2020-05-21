# Rap Lyric Generator
I built a model which given some lyrics as input can generate some decent bars of its own. The project was inspired by a [similar project](https://github.com/nikolaevra/drake-lyric-generator). The model is word-level (given a sequence of words as input it predicts the next, most likely word), uses 2 LSTM layers, and makes use of word embedding. 

I trained the model on the songs of some of my favourite rap artists to yield results which were similar to the songs I liked. In the end, the results were pretty good. It was impressive how the model learned to form coherent lines, have successive lines relate in context, insert newlines at appropriate times, and at times follow a theme.

I also trained a second model, which was the same as the one mentioned above however with two bidirectional LSTMs rather than two conventional ones. I thought it might improve the models ability to pick on context, themes, and have successive lines flow better. In the end, there was a slight improvement in the quality of individual lines and the model produced good results more consistently. However, there wasn't a notable improvement in the areas I had expected. 

## Contents
* [Files (and Usage)](#files-and-usage)
* [Data Preprocessing](#data-preprocessing)
    * [Develop word embeddings (model.py / train.py):](#develop-word-embeddings-modelpy--trainpy)
* [Model](#model)
* [Model Training](#model-training)
* [Model Evaluation](#model-evaluation)
* [Model Results](#model-results)
    * [Conventional LSTM Model](#conventional-lstm-model)
    * [Bidirectional LSTM Model](#bidirectional-lstm-model)
* [Aside: Looking at the Impact of Embedding Dimensions on Model Performance](#aside-looking-at-the-impact-of-embedding-dimensions-on-model-performance)


## Files (and Usage)
1. `data/data_scraper.py` 
    - a simple web scraper to generate the dataset `songs.csv` given `artist_list.csv` using data from [metrolyrics.com](https://www.metrolyrics.com/)
2. `data/process_data.py` 
    - cleans and encodes the scraped songs
    - splits the songs into training+validations and test sets, and dumps as text files
    - creates the vocab set, the word to index mapping lookup table, the index to word mapping lookup table, and dumps these files
    - prints some interesting metrics (ex. top 50 and bottom 50 words in the dataset in terms of occurrence)
3. `model.py`
    - contains the model definition(s)
4. `train.py` 
    - splits the encoded training set into training examples
    - trains the model on the generated training examples using a model defined in `model.py`
    - requires the dumped encoded training set, vocab set, and lookup tables
5. `test.py [--h5] [--ckpt]`  
    - loads a model saved as an `.h5` file or loads the weights from the most recent `.ckpt` file in the `results/` directory into the appropriate model definition from `model.py`
    - requires the dumped test set, vocab set, and lookup tables
6. `rap-lyric-generator-training.ipynb`
    - used this for training, shouts to Google Collab for the free GPUs
7. `effect-of-embedding-dimension.ipynb`
    - wanted to look at the effect of embedding dimension size on model performance, see *Aside*

## Data Preprocessing

I scraped the songs I used from [metrolyrics.com](https://www.metrolyrics.com/). I did the following to each song to yield a dataset better suited for the learning task.

**Get and clean the song lyrics (`data_scraper.py`):**

- Remove all parentheses and brackets along with their contents
- Make all letters lowercase
- Remove all punctuation
- Replace newlines `\n` with `|-|` so that each song is a single line
- Remove all other string literals
- Shuffle the songs in the dataset

**Encode each word (`process_data.py`):**

- Create a vocabulary set consisting of all words in the training + validation set which occur at least 10 times (idea taken from this [paper](https://arxiv.org/pdf/1901.09785.pdf)) and are at least of length 2 (unless to the word is "i" or "a")
- Replace each word with the index of the word in the vocabulary set, all words that don't exist in the vocab set are encoded with the value of the last index + 1 (i.e. `len(vocabSet)`)

**Create training examples (`train.py`):**

- Split the songs into a set of sequences of length = `model.seq_length` to be used as input and a corresponding set of target words to be used as the desired output
- Save the training examples as a TensorFlow `dataset.Dataset` object to allow for large datasets without memory issues ([see](https://www.tensorflow.org/api_docs/python/tf/data/Dataset))
- Shuffle the training examples
- Batch the training examples for faster training using batch size = `model.batch_size`

```
total # of songs:  3448
# of songs in train, val, and test sets respectively: 2069, 690, 689

total # of words:  1718107
# of unique words in entire set:  35652
size of vocabSet:  7016
```

### Develop word embeddings (`model.py` / `train.py`):

Word embeddings are used to represent words in a dense vector form which is more informative. One-hot encoding - a common representation method of words for NLP applications - creates extremely sparse vector representations of each word in the vocabulary set and contains no information about the relation between words. An embedding layer (a neural network of its own) attempts to learn dense vector representation of each word in the vocab set such that the vectors of two related words will have a higher cosine similarity (minimizing the angle between the two vectors). In other words, the more similar two words are in context, the more similar their embedding vector *should* be. 

The dimensions of an embedding vector is a parameter of the embedding layer. If one chooses to set the embedding dimensions to 64, the layer will create a dense vector representation of each word that is of length 64. This creates the added benefit of keeping the vector representation of each word relatively small in length even with extremely large vocab sets when compared to one-hot encoding. 

## Model

I created two versions of the model: one using conventional LSTMs and another using bidirectional LSTMs. For both I started with an embedding layer setting the embedding dimensions to 256. Following this, I used two LSTM layers setting each of their RNN units to 512 (took the idea from this [paper](https://pdfs.semanticscholar.org/c51d/13034b2df47dae8f33bd0efad996de99ed4c.pdf)). I then had a Dense layer at the end using a softmax activation to output the probability of each word in my vocab set being the target word. 

*Thanks TensorFlow Keras for making this easy.* 

## Model Training

For my models specifically I used a batch size of 128 and trained the model over 50 epochs using `train.py`. I trained the model on 1,443,915 training examples generated from the encoded training + validation set. 

## Model Evaluation

As the model is being used for the task of text generation, it is difficult to attach a numerical score to the results as the strength of a prediction is largely subjective. Below are some select good and not so good model outputs.

## Model Results

The following outputs were generated by feeding inputs from the encoded test set. The results were post-processed to replace `|-|` outputs predicted by the model with `\n`. The lyrics are not censored so read at your own discretion. 

### Conventional LSTM Model

Some good results:

```
INPUT LINES:
door shuts in your face promise me if i cave in and break
and leave myself open that i wont be makin a mistake
cause im a
im a space bound rocket ship and your hearts the moon
and im aimin right at you right at you
250 000 miles on a clear night in june
and im aimin right at you right at you right at you
so after a year and six months its no longer me that you want
but i love you so much it hurts never mistreated

PREDICTED NEXT LINES:
me 
or maybe nothin else you tryin to say 
but i aint never did a reason for you 
but when you get to see me all up lookin at me 
i dont wanna talk without a few 
you aint never seen a body 
you say you fucked me 
feelin like the beat is the drugs ive been through 
i cant sleep i gotta be your kid today 
this has been years i cant have it 
ive been kicked the world to run away 
i think im tougher than most
```

```
INPUT LINES:
thats upsidedown cake
get em a red nose they clown cakes
they shoulda never let you round cake
look at my neck i got a carrot cake
now heres the icing on the cake
cake cakecake cakecake
uh im just gettin started oh yeah we got it bitch
ive done made more millionaires than the lotto did
dame made millions bigg made millions
ye made millions just made millions
lyor made millions cam made millions
beans tell you if he wasnt in his feelings
im back in my

PREDICTED NEXT LINES:
cloud aint no love reminiscing 
picture me in the struggle 
tho pass the weed in the air 
oh what a feeling 
whatever you want 
feel the walk in the world til you last 
where brooklyn at where the west coast 
now im the first time i was 21 
now im back in the town 
i was hidin game on my brain 
i was broke i was a star 
tip or face and my momma told me nas 
i was broke i was grip with death 
hustling like
```

```
INPUT LINES:
said do you wanna do you wanna be free
free from pain free from scars
free to sing free from bars
free my dawgs youre free to go
block gets hot the streets is cold
free to love to each his own
free from bills free from pills
you roll it loud the speakers blow
life get hard it eats your soul
it clears your mind learn to fly
then reach the stars you take your time
and look behind and said look where i came
look how

PREDICTED NEXT LINES:
youve seen somethin 
so when you talk to me 
see you love me 
sometimes i feel like when your father saw 
and i blame what i did gonna steal you 
i was gonna kill you and i knew you 
and i dont lack shit 
and i dont lack shit 
who you are 
enough pressure to say goodbye to me 
you can call me a thousand miles 
im in the midst of the lies 
ive been waiting for you 
alright 
hey 
shady aftermath ever focused
```

Bad result:

```
INPUT LINES:
it big like costco
and we back in this bitch like a tampon
still fuck dirty ems clean like a bar o soap
and you were so slick on some baffoon shit
my hands on a sweeper your was on a broomstick
stop lookin all stupid im rude and abusive
and strapped dont make me use it
chorus
hit me with your best shot hit me with your best shot
hit me with your best shot hit me with your best shot
hit me with your best shot hit me

PREDICTED NEXT LINES:
your bitch red 
you cant see me compton 
while you thinkin about me 
you nasty 
you nasty 
you know i gotta go 
let me off your dick 
and let it go 
take everything 
you ready to breathe 
dont go somewhere inside your mouth 
where you at 
where you at 
where you at 
where you at 
where you at 
where you at 
where you at 
where you at 
where you at 
where you at 
where you at 
where you
```

### Bidirectional LSTM Model

Some good results:

```
INPUT LINES:
pay like you weigh wet wit two pairs of clothes on
now get yo ass to the back as im flyin to the track
timbaland let me spit my prose on
pump it up in the prozone
thats the track that we breakin these hoes on
aint the track that we flows on
but when shit get hot then the glock start poppin like ozone
we keep hoes crunk like triggerman
fo real it dont get no bigger man
dont trip lets flip gettin throwed on the flip
gettin blowed

PREDICTED NEXT LINES:
nigga sit down and get paid 
get up out of shit wit all these killas 
while im in hell i know i still heard knife 
and dont you ever come to see fame and still dont do it 
if you aint talkin bout game you fucking with none 
and they dont like to lay them back like they fathers 
i know they scared to die bitch im just a man 
ill be at my one too much but ill sell tuition 
i might be the atl to the game 
and imma
```

```
INPUT LINES:
|-| no looking back when im gone
no more
no more
no more
no more
no more
no more
no more
no more
no more
no more
no more
no more
no more
no more
no more
no more
its over yeah it over yeah im leaving im gone
ive been doing this wrong ive been here for too long
yeah im leaving im leave you know i got my reasons
yeah im leaving yeah im leaving im leaving im

PREDICTED NEXT LINES:
still here for granted 
im feelin alive 
im out for years left the phone 
im just tryna get it 
im in love with you 
im a ho im a nice one 
yeah its back to you 
but everything im not made me everything 
im doin it 
really im the best 
still trying to hold you gonna be right 
and baby im mad at you look at that 
yea baby im on 
my way im feelin me 
ive been down so deep in my heart 
i
```

Bad result:

```
INPUT LINES:
house into a happy home
happy home
heh
|-| going through these hard times
ghetto child of mine
i wonder if you have to suffer for your fathers crimes
to be honest its a hard road
just keep your faith in god knowing youll get scarred though
look at him walking and talking a little child with my eyes and mouth
father watch over little seeds help me guide them out
had to change my whole lifestyle
married my babys mama made her my wife now
im trying

PREDICTED NEXT LINES:
to get off my mind 
im out my face 
why you act 
all you do is 
wonder if your mamas get cheatin 
or give me a promise 
catch me dont you 
catch me with a bottle 
keep a black picture bag 

side 
now everybody in the old x2 

times up 
come on get up 
young niggas ballin 
huh pull the fuck up 
watch ya mouth 
you pop a bottle 
your motherfuckin dick 
keep it up 
keep it
```

## Aside: Looking at the Impact of Embedding Dimensions on Model Performance

I was curious at how the chosen embedding dimensions would affect the performance of an LSTM model. I created a simple model and trained it using a set of different embedding dimensions, 150,000 training examples, and over 15 epochs.

```python
model = [
	Embedding(embedding_dims=[64, 128, 256, 512, 1024]),
	LSTM(rnn_units=512),
	Dense(activation='softmax', output_length=vocabSet_size),
	]
```

I then computed the accuracy and cross entropy of each of the trained models on the validation set (50,000 examples). This produced the following graphs.

![Accuracy Graph](/results/embedding_dim_effect_accuracy.png?raw=true)

![Cross Entropy Graph](/results/embedding_dim_effect_crossentropy.png?raw=true)

Now, admittedly these are flawed metrics for my task. I don't need the model to predict exactly the right word each time as there is typically no word which is "exactly right", but I want to make sure their context is the same. Instead, I could compare the learned embedding representation of the predicted and the target word as they should be relatively close, maybe using mean-squared error. But that requires some level of work that I don't feel like doing (maybe in the future, for now this project has taken long enough :^) ). However, this is still not a perfect solution as the embedding layer may incorrectly learn some words to be similar even if they are not. So, one would have to evaluate the effectiveness of their embedding layer first and take this performance into consideration. 

Nevertheless, I found my results interesting. Particularly that although accuracy increased with embedding dimension size (albeit, still pretty low), cross entropy also increased. One would expect the opposite as cross entropy heavily penalizes when the model outputs a small probability for the target class. [This post](https://stats.stackexchange.com/questions/258166/good-accuracy-despite-high-loss-value) talks about how the sweet spot for accuracy does not always occur at the minimum cross entropy. Something to delve deeper into in the future.

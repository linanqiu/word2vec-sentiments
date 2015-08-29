
# Sentiment Analysis using Doc2Vec

Word2Vec is dope. In short, it takes in a corpus, and churns out vectors for each of those words. What's so special about these vectors you ask? Well, similar words are near each other. Furthermore, these vectors represent how we use the words. For example, `v_man - v_woman` is approximately equal to `v_king - v_queen`, illustrating the relationship that "man is to woman as king is to queen". This process, in NLP voodoo, is called **word embedding**. These representations have been applied widely. This is made even more awesome with the introduction of Doc2Vec that represents not only words, but entire sentences and documents. Imagine being able to represent an entire sentence using a fixed-length vector and proceeding to run all your standard classification algorithms. Isn't that amazing?

However, Word2Vec documentation is shit. The C-code is nigh unreadable (700 lines of highly optimized, and sometimes weirdly optimized code). I personally spent a lot of time untangling Doc2Vec and crashing into ~50% accuracies due to implementation mistakes. This tutorial aims to help other users get off the ground using Word2Vec for their own research. We use Word2Vec for **sentiment analysis** by attempting to classify the Cornell IMDB movie review corpus (http://www.cs.cornell.edu/people/pabo/movie-review-data/).

The source code used in this demo can be found at https://github.com/linanqiu/word2vec-sentiments

## Setup

### Modules

We use `gensim`, since `gensim` has a much more readable implementation of Word2Vec (and Doc2Vec). Bless those guys. We also use `numpy` for general array manipulation, and `sklearn` for Logistic Regression classifier.

```python
# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression
```

### Input Format

We can't input the raw reviews from the Cornell movie review data repository. Instead, we clean them up by converting everything to lower case and removing punctuation. I did this via bash, and you can do this easily via Python, JS, or your favorite poison. This step is trivial.

The result is to have five documents:

- `test-neg.txt`: 12500 negative movie reviews from the test data
- `test-pos.txt`: 12500 positive movie reviews from the test data
- `train-neg.txt`: 12500 negative movie reviews from the training data
- `train-pos.txt`: 12500 positive movie reviews from the training data
- `train-unsup.txt`: 50000 Unlabelled movie reviews

Each of the reviews should be formatted as such:

```
once again mr costner has dragged out a movie for far longer than necessary aside from the terrific sea rescue sequences of which there are very few i just did not care about any of the characters most of us have ghosts in the closet and costner s character are realized early on and then forgotten until much later by which time i did not care the character we should really care about is a very cocky overconfident ashton kutcher the problem is he comes off as kid who thinks he s better than anyone else around him and shows no signs of a cluttered closet his only obstacle appears to be winning over costner finally when we are well past the half way point of this stinker costner tells us all about kutcher s ghosts we are told why kutcher is driven to be the best with no prior inkling or foreshadowing no magic here it was all i could do to keep from turning it off an hour in
this is an example of why the majority of action films are the same generic and boring there s really nothing worth watching here a complete waste of the then barely tapped talents of ice t and ice cube who ve each proven many times over that they are capable of acting and acting well don t bother with this one go see new jack city ricochet or watch new york undercover for ice t or boyz n the hood higher learning or friday for ice cube and see the real deal ice t s horribly cliched dialogue alone makes this film grate at the teeth and i m still wondering what the heck bill paxton was doing in this film and why the heck does he always play the exact same character from aliens onward every film i ve seen with bill paxton has him playing the exact same irritating character and at least in aliens his character died which made it somewhat gratifying overall this is second rate action trash there are countless better films to see and if you really want to see this one watch judgement night which is practically a carbon copy but has better acting and a better script the only thing that made this at all worth watching was a decent hand on the camera the cinematography was almost refreshing which comes close to making up for the horrible film itself but not quite
```

The sample up there contains two movie reviews, each one taking up one entire line. Yes, **each document should be on one line, separated by new lines**. This is extremely important, because our parser depends on this to identify sentences.

### Feeding Data to Doc2Vec

Doc2Vec (the portion of `gensim` that implements the Doc2Vec algorithm) does a great job at word embedding, but a terrible job at reading in files. It only takes in `LabeledLineSentence` classes which basically yields `LabeledSentence`, a class from `gensim.models.doc2vec` representing a single sentence. Why the "Labeled" word? Well, here's how Doc2Vec differs from Word2Vec.

Word2Vec simply converts a word into a vector.

Doc2Vec not only does that, but also aggregates all the words in a sentence into a vector. To do that, it simply treats a sentence label as a special word, and does some voodoo on that special word. Hence, that special word is a label for a sentence. 

So we have to format sentences into

```python
[['word1', 'word2', 'word3', 'lastword'], ['label1']]
```

`LabeledSentence` is simply a tidier way to do that. It contains a list of words, and a label for the sentence. We don't really need to care about how `LabeledSentence` works exactly, we just have to know that it stores those two things -- a list of words and a label.

However, we need a way to convert our new line separated corpus into a collection of `LabeledSentence`s. The default constructor for the default `LabeledLineSentence` class in Doc2Vec can do that for a single text file, but can't do that for multiple files. In classification tasks however, we usually deal with multiple documents (test, training, positive, negative etc). Ain't that annoying?

So we write our own `LabeledLineSentence` class. The constructor takes in a dictionary that defines the files to read and the label prefixes sentences from that document should take on. Then, Doc2Vec can either read the collection directly via the iterator, or we can access the array directly. We also need a function to return a permutated version of the array of `LabeledSentence`s. We'll see why later on.

```python
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        return numpy.random.permutation(self.sentences)
```

Now we can feed the data files to `LabeledLineSentence`. As we mentioned earlier, `LabeledLineSentence` simply takes a dictionary with keys as the file names and values the special prefixes for sentences from that document. The prefixes need to be unique, so that there is no ambiguitiy for sentences from different documents.

The prefixes will have a counter appended to them to label individual sentences in the documetns.

```python
sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

sentences = LabeledLineSentence(sources)
```

## Model

### Building the Vocabulary Table

Doc2Vec requires us to build the vocabulary table (simply digesting all the words and filtering out the unique words, and doing some basic counts on them). So we feed it the array of sentences. `model.build_vocab` takes an array of `LabeledLineSentence`, hence our `to_array` function in the `LabeledLineSentences` class. 

If you're curious about the parameters, do read the Word2Vec documentation. Otherwise, here's a quick rundown:

- `min_count`: ignore all words with total frequency lower than this. You have to set this to 1, since the sentence labels only appear once. Setting it any higher than 1 will miss out on the sentences.
- `window`: the maximum distance between the current and predicted word within a sentence. Word2Vec uses a skip-gram model, and this is simply the window size of the skip-gram model.
- `size`: dimensionality of the feature vectors in output. 100 is a good number. If you're extreme, you can go up to around 400.
- `sample`: threshold for configuring which higher-frequency words are randomly downsampled
- `workers`: use this many worker threads to train the model 

```python
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

model.build_vocab(sentences.to_array())
```

### Training Doc2Vec

Now we train the model. The model is better trained if **in each training epoch, the sequence of sentences fed to the model is randomized**. This is important: missing out on this steps gives you really shitty results. This is the reason for the `sentences_perm` method in our `LabeledLineSentences` class.

We train it for 10 epochs. If I had more time, I'd have done 20.

This process takes around 10 mins, so go grab some coffee.

```python
for epoch in range(10):
    model.train(sentences.sentences_perm())
```

### Inspecting the Model

Let's see what our model gives. It seems that it has kind of understood the word `good`, since the most similar words to good are `glamorous`, `spectacular`, `astounding` etc. This is really awesome (and important), since we are doing sentiment analysis.

```python
model.most_similar('good')

[(u'tekashi', 0.45127424597740173),
 (u'glamorous', 0.4344240427017212),
 (u'spectacular', 0.42718690633773804),
 (u'astounding', 0.42001062631607056),
 (u'valentinov', 0.41705751419067383),
 (u'sweetest', 0.4043062925338745),
 (u'complementary', 0.4039931297302246),
 (u'boyyyyy', 0.39713743329048157),
 (u'macdonaldsland', 0.3965899348258972),
 (u'elven', 0.39042729139328003)]
```


We can also prop the hood open and see what the model actually contains. This is each of the vectors of the words and sentences in the model. We can access all of them using `model.syn0` (for the geekier ones among you, `syn0` is simply the output layer of the shallow neural network). However, we don't want to use the entire `syn0` since that contains the vectors for the words as well, but we are only interested in the ones for sentences.

Here's a sample vector for the first sentence in the training set for negative reviews:

```python
model['TRAIN_NEG_0']

array([ 0.45238438, -0.07346677, -0.17444436,  0.60655016, -0.70522565,
        0.28476399,  0.24404588,  0.09271102, -0.02715847, -0.13526627,
       -0.12390804, -0.00219905,  0.011253  ,  0.24557671, -0.09958933,
        0.17554867,  0.16079453, -0.18499082, -0.31598854,  0.01447532,
        0.52194822, -0.2387463 ,  0.16799606,  0.47053325,  0.09696233,
       -0.2582404 , -0.19224562, -0.07114315, -0.25864932, -0.5387702 ,
        0.01053433,  0.43367237,  0.07885301,  0.04634216,  0.0899957 ,
        0.06260718, -0.38053334,  0.18118465,  0.14301547,  0.18286002,
       -0.31105465,  0.2040111 , -0.76622951,  0.06977512,  0.11759907,
       -0.11566088, -0.00373716, -0.14705311, -0.29019266, -0.04825564,
        0.20127594, -0.0258627 , -0.20973501,  0.48925173, -0.31426486,
        0.3180953 ,  0.41300809, -0.29024398, -0.21187432,  0.10730035,
        0.30392009, -0.2130826 ,  0.47062019, -0.17570473,  0.21256927,
        0.51417089, -0.00951673,  0.1525774 ,  0.05895659,  0.33289343,
        0.56261861, -0.05355176, -0.05011608,  0.24092411, -0.17943399,
       -0.26373053,  0.22000515,  0.05890461, -0.24378468,  0.58705276,
        0.01776701,  0.04332061, -0.04941204,  0.24699709, -0.28202724,
       -0.27278683,  0.2515423 ,  0.12944862, -0.29060578, -0.02939321,
        0.42860341, -0.27076352, -0.56153166,  0.35900518, -0.11538842,
       -0.29707447,  0.15181458,  0.73098952,  0.308236  ,  0.52810729], dtype=float32)
```


### Saving and Loading Models

To avoid training the model again, we can save it.

```python
model.save('./imdb.d2v')
```

And load it.

```python
model = Doc2Vec.load('./imdb.d2v')
```

## Classifying Sentiments

### Training Vectors

Now let's use these vectors to train a classifier. First, we must extract the training vectors. Remember that we have a total of 25000 training reviews, with equal numbers of positive and negative ones (12500 positive, 12500 negative).

Hence, we create a `numpy` array (since the classifier we use only takes numpy arrays. There are two parallel arrays, one containing the vectors (`train_arrays`) and the other containing the labels (`train_labels`).

We simply put the positive ones at the first half of the array, and the negative ones at the second half.

```python
train_arrays = numpy.zeros((25000, 100))
train_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model[prefix_train_pos]
    train_arrays[12500 + i] = model[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0
```

The training array looks like this: rows and rows of vectors representing each sentence.

```python
print train_arrays

[[ 0.42028627 -0.0910796  -0.10316094 ..., -0.11574443  0.54547763
  -0.1086079 ]
 [ 0.21860494  0.34468749 -0.06821636 ..., -0.02118306  0.39692196
   0.518085  ]
 [ 0.19905667 -0.05517581  0.0789782  ...,  0.78548694  0.10369277
   0.15604787]
 ..., 
 [ 0.42894334 -0.03023763 -0.38231012 ...,  0.17735066  0.36474037
  -0.08756389]
 [ 0.65340477  0.388024   -0.34454256 ...,  0.0466847   0.61409295
   0.19534792]
 [ 0.40329584  0.26531416 -0.11242788 ...,  0.08738184  0.48685795
  -0.17476116]]
```

The labels are simply category labels for the sentence vectors -- 1 representing positive and 0 for negative.

```python
print train_labels

[ 1.  1.  1. ...,  0.  0.  0.]
```

### Testing Vectors

We do the same for testing data -- data that we are going to feed to the classifier after we've trained it using the training data. This allows us to evaluate our results. The process is pretty much the same as extracting the results for the training data.

```python
test_arrays = numpy.zeros((25000, 100))
test_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model[prefix_test_pos]
    test_arrays[12500 + i] = model[prefix_test_neg]
    test_labels[i] = 1
    test_labels[12500 + i] = 0
```

### Classification

Now we train a logistic regression classifier using the training data.

```python
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
```

And find that we have achieved near 87% accuracy for sentiment analysis. This is rather incredible, given that we are only using a linear SVM and a very shallow neural network.

```
classifier.score(test_arrays, test_labels)

0.86968000000000001
```


Isn't this fantastic? Hope I saved you some time!

## References

- Doc2vec: https://radimrehurek.com/gensim/models/doc2vec.html
- Paper that inspired this: http://arxiv.org/abs/1405.4053

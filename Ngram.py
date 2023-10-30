from collections import defaultdict, Counter
import random, math

UNK = '<UNK>'
START = '<START>'
END = '<END>'
idx2word = {0: UNK, 1: START, 2: END}
word2idx = {UNK: 0, START: 1, END: 2}

class Ngram(object):
    def __init__(self, n, train_data):
        self.n = n # n-gram order
        self.train_data = train_data # sentences for training
        self.ngram_counts = defaultdict(Counter) # counts of ngram types
        self.ngram_probs = defaultdict(Counter) # probabilities of ngram types
        
        # calculate counts and probabilities
        self.buildCounts()
        self.vocab_size = self.getVocabSize()
        self.buildProbabilities()
        
    def padStartSentence(self, sentence):
        return [word2idx[START]]*(self.n-1) + sentence[1:]
    
    def trimStartSentence(self, sentence):
        return [word2idx[START]] + sentence[self.n-1:]
        
    def buildCounts(self):
        for sentence in self.train_data:
            sentence = self.padStartSentence(sentence)
            for i in range(self.n - 1, len(sentence)):
                ngram_prefix = tuple(sentence[i - self.n + 1 : i])
                ngram_suffix = tuple([sentence[i]])
                self.ngram_counts[ngram_prefix][ngram_suffix] += 1
                
    def getVocabSize(self):
        V = 0
        for suffixes in self.ngram_counts.values():
            V += sum(suffixes.values())
        return V
                
    def calculateProbability(self, ngram):
        ngram_prefix = tuple(ngram[:-1])
        ngram_suffix = tuple([ngram[-1]])
        if sum(self.ngram_counts[ngram_prefix].values()) > 0:
            return self.ngram_counts[ngram_prefix][ngram_suffix] / sum(self.ngram_counts[ngram_prefix].values())
        else:
            return 0

    def buildProbabilities(self):
        for ngram_prefix, ngram_suffixes in self.ngram_counts.items():
            for ngram_suffix in ngram_suffixes.keys():
                self.ngram_probs[ngram_prefix][ngram_suffix] = self.calculateProbability(ngram_prefix + ngram_suffix)
    
    def generateSentence(self):
        sentence = self.padStartSentence([])
        continue_sentence = True
        
        while continue_sentence:
            ngram_prefix = tuple(sentence[-self.n+1:]) if self.n > 1 else tuple([])
            ngram_suffix = random.choices(
                list(self.ngram_probs[ngram_prefix].keys()), list(self.ngram_probs[ngram_prefix].values())
            )[0][0]
            sentence.append(ngram_suffix)
            continue_sentence = True if ngram_suffix != word2idx[END] else False
            
        return self.trimStartSentence(sentence)
    
    def getSentenceLogLikelihood(self, sentence):
        sentence = self.padStartSentence(sentence)
        probabilities = [ self.calculateProbability(sentence[i - self.n + 1 : i + 1]) for i in range(self.n - 1, len(sentence)) ]
        return sum([math.log(p) for p in probabilities])
    
    def getCorpusPerplexity(self, test_data):
        sum_loglik, ngram_count = 0, 0
        
        for sentence in test_data:
            sum_loglik += -self.getSentenceLogLikelihood(sentence)
            ngram_count += len(self.padStartSentence(sentence)) - (self.n - 1)
            
        return math.exp(sum_loglik / ngram_count)
    
    def getTopN(self, n):
        ngrams = defaultdict(int)
        for ngram_prefix, suffixes in self.ngram_counts.items():
            for ngram_suffix, count in suffixes.items():
                ngrams[ngram_prefix + ngram_suffix] += count
        sorted_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
        return sorted_ngrams[:n]
    
#### AddAlphaSmooth class ####
class AddAlphaSmooth(Ngram):
    def __init__(self, n, train_data, alpha=1):
        self.n = n # n-gram order
        self.alpha = alpha
        self.train_data = train_data # sentences for training
        self.ngram_counts = defaultdict(Counter) # counts of ngram types
        self.ngram_probs = defaultdict(Counter) # probabilities of ngram types
        
        # calculate counts and probabilities
        self.buildCounts()
        self.vocab_size = self.getVocabSize()
        self.buildProbabilities()
    
    def calculateProbability(self, ngram):
        ngram_prefix = tuple(ngram[:-1])
        ngram_suffix = tuple([ngram[-1]])
        if sum(self.ngram_counts[ngram_prefix].values()) > 0:
            return (self.ngram_counts[ngram_prefix][ngram_suffix] + self.alpha) / (sum(self.ngram_counts[ngram_prefix].values()) + len(self.ngram_counts[ngram_prefix].values()))
        else:
            return self.alpha / (sum(self.ngram_counts[ngram_prefix].values()) + self.alpha*self.vocab_size)
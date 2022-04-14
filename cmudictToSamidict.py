
            #print(currentNode.word, end)
            #if '_' in currentNode.word and (word[end] in currentNode.children and
            #    currentNode.children[word[end]].end): totalMatches.add((start, end, currentNode.children[word[end]].word))

from itertools import combinations
import json
from operator import indexOf
from nltk.corpus import cmudict
import ahocorasick
import re
from ast import literal_eval
from argparse import ArgumentParser
graphemeToPhoneme = json.load(open("graphemeToPhoneme.json"))
parser = ArgumentParser()
parser.add_argument('--words', nargs='+')
args = parser.parse_args()
print(args.words)
class Trie:
    def __init__(self, root=None, word="", output = set()) -> None:
        self.root = root
        self.children = {}
        self.end = False
        self.word = word 
        self.failLinks = {}
        self.output = output

    def updateAllOutput(self, wordlist = set()):
        #if word == 'a':
        #    print(list(self.children[child].word for child in self.children))
        #print(self.word, self.output, wordlist)
        if self.end: wordlist.add(self.word)
        self.output = self.output.union(wordlist)
        for child in self.children:
            self.children[child].updateAllOutput(set(wordlist))

    def addWord(self, word):
        if word:
            self.children[word[0]] = ( 
                next((t[1] for t in self.children.items() if t[0] == word[0]), 
                    Trie(self.root if self.root else self
                    , self.word + word[0], output=set(x for x in self.output if x[0] == word[0]))).addWord(word[1:]))
        else: 
            self.end = True
            self.output.add(self.word)
        return self

    def printAllWords(self):
        if self.end: print(self.word, self.output)
        for key in self.children:
            self.children[key].printAllWords()
    
    def setFailState(self):
        if not self.root: return None
        self.failLinks = self.root.findFailState(self.word)
        #print(self.word, dict((x[0], x[1].word) for x in self.failLinks.items()))
    
    def setAllFailStates(self):
        self.setFailState()
        for child in self.children:
            self.children[child].setAllFailStates()
    
    def findFailState(self, word):
        '''Should only be called by root Node'''
        
        assert self.root == None
        for l in range(1, len(word)):
            idx = l
            currentTries = {'': self}
            while idx < len(word):
                if word[idx] == '_':
                    #print(word, list(currentTrie.children[w].word for currentTrie in currentTries.values() for w in currentTrie.children))
                    newDict = dict((w, currentTrie.children[word[idx - 1]].children[w]) 
                                             for currentTrie in currentTries.values() for w in currentTrie.children if w in currentTrie.children.get(word[idx - 1], Trie()).children)
                    currentTries = dict((w, currentTrie.children[w]) for currentTrie in currentTries.values() for w in currentTrie.children)                    
                    currentTries.update(newDict)
                else: 
                    for i, currentTrie in currentTries.items():
                        if word[idx] in currentTrie.children:
                            currentTries[i] = currentTrie.children[word[idx]]  
                            #print(idx, currentTries[i].word + " Success for " + word)
                        elif i != '':
                            #print(idx, currentTries[i].word + " Failed for " + word)
                            currentTries[i] = currentTrie.root.children[i]
                        else:
                            currentTries[i] = None
                currentTries = dict(filter(lambda x: x[1], currentTries.items()))
                if not currentTries: break
                idx += 1
            if idx >= len(word):
                #print(word, list(currentTries.items())[0][0])
                return currentTries
        return {'': self}

    def __repr__(self) -> str:
        return json.dumps({letter.upper() if child.end else letter.lower(): json.loads(str(child)) for letter, child in self.children.items()}, indent=4)

    def findAllMatches(self, word):
        assert self.root == None
        totalMatches = set()
        currentNode = self
        start, end = 0, 0
        
        #Check against every end letter
        while end < len(word) or len(currentNode.word) > 1:
            #print(end, sorted(totalMatches))
            #If Next Letter Can Be A Blank
            if end + 1 < len(word) and "_" in currentNode.children:
                #Go To Blank First
                currentNode = currentNode.children['_']
                end += 1 
            #If Next Letter Exists in Word Children
            elif end < len(word) and word[end] in currentNode.children:
                #Traverse to Node
                currentNode = currentNode.children[word[end]]
                if currentNode.end: totalMatches.add((end - len(currentNode.word) + 1, end, currentNode.word))
                end +=1
            else:
                if ((currentNode.word != "" and word[start:end + 1] != currentNode.word) or 
                        ((not end < len(word)) and len(currentNode.word) > 1)):
                    start  +=1
                    print(start, currentNode.word)
                    if '_' in currentNode.word:                       
                        if indexOf(currentNode.word, '_') == 0:
                            newNode = currentNode.root.children[word[end - len(currentNode.word)]]
                            start -= 1
                        else:
                            newNode = currentNode.failLinks.get(word[end + indexOf(currentNode.word, '_') - len(currentNode.word)], 
                                                                next(iter(currentNode.failLinks.values())))  
                            if len(currentNode.word) == len(newNode.word):
                                start -= 1   
                        currentNode = newNode                            
                    elif start != end and (word[start] in currentNode.failLinks):
                        currentNode = currentNode.failLinks[word[start + 1]]
                    else: 
                        currentNode = currentNode.failLinks['']  
                    totalMatches = totalMatches.union((start, start + len(w) - 1, w) for w in currentNode.output)  
                    end = start + len(currentNode.word)  
                      
        return(totalMatches)

def update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and k in d:
            d[k] = update(d[k], v)
        else: d[k] = v
    return d

def createGraphemeList(words, listofGrapheme):
    #word[0] == start, word[1] == end, word[2] == word
    #print(words)
    return dict(map(
        lambda word: 
            #If word does not have "_" check if start of a grapheme matches with end of current grapheme, 
            (word[2], 
                createGraphemeList(
                    list(filter(lambda x: x[0] == word[1] + 1, listofGrapheme)), 
                        listofGrapheme)) if "_" not in word[2] else (
            #If word does have "_", additionally check if start and end of grapheme cover hole of "_"
            word[2], {next(filter(lambda x: x[0] == word[0] + indexOf(word[2], "_") and x[1] == word[0] + indexOf(word[2], "_"), listofGrapheme))[2]:
                createGraphemeList(
                    list(filter(lambda x: x[0] == word[1] + 1, listofGrapheme)), 
                        listofGrapheme)
            }), 
        words)) 

def flatten(phonemeList, word):
    to_add = []
    for phoneme in phonemeList:
        if phoneme == '$': to_add.append(word)
        else: to_add.extend(tuple(flatten(phonemeList[phoneme], (*word, phoneme))))
    return to_add        

def createPhomemeList(graphemeList):
    phonemeList = {}
    if not graphemeList: return {"$": None}
    for grapheme in graphemeList:
        for phoneme in graphemeToPhoneme[grapheme]:
            if isinstance(phoneme, list):
                phonemeList[phoneme[0]] = update(phonemeList.get(phoneme[0], {}), {
                    phoneme[1]: createPhomemeList(graphemeList[grapheme])
                })
            else: phonemeList[phoneme] = update(phonemeList.get(phoneme, {}), createPhomemeList(graphemeList[grapheme]))
    return phonemeList

def substring(word):
    for i, phoneme in enumerate(word):
        word[i] = re.sub('\d+', '', phoneme)

    for i, j in combinations(range(len(word) + 1), 2):
        yield tuple(word[i:j])

def reverseDictionary(d):
    
    rDict = ((substring(p), word) for word in d for p in d[word])
    result = {}
    for t in rDict:
        for s in t[0]:
            result.setdefault(s, []).append(t[1])
    return result

if __name__ == "__main__":
    trie = Trie()
    for word in graphemeToPhoneme:
        trie.addWord(word) 
    trie.updateAllOutput()
    trie.setAllFailStates()
    dictionary = json.load(open("reversedDictionary.json"))
    print("ready")
    for i in args.words:  
        if not i.isalpha(): continue
        result = trie.findAllMatches(i)
        #print(sorted(result))
        graphemeList = createGraphemeList(list(filter(lambda x: x[0] == 0, result)), result)
        #print(graphemeList)
        phonemeList = flatten(createPhomemeList(graphemeList), '')
        #print(phonemeList)
        #p = next(filter(lambda x: str(x) in dictionary, phonemeList))
        #print(p)
        finalList = sorted(list((str(k), str(dictionary[str(k)])) for k in filter(lambda x: str(x) in dictionary, phonemeList)), 
                           key=lambda x: (len(list(filter(lambda word: i in word, x[1])))) / len(x[1]))
        print(json.dumps(finalList, indent=2))
        #if any(filter(lambda words: i in word, (d[1] for d in finalList))):
        #    print(i)
    
    #for idx, i in enumerate(list(json.load(open("samiDict.json")).keys())):
             

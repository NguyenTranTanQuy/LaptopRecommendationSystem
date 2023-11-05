import os
import pandas as pd
import gensim
import numpy as np
from DataPreprocessing import preprocess_text

current_file = __file__
f = os.path.dirname(os.path.abspath(current_file))

class IncrementalLearning:
    def __int__(self, filePath):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(f + '/Word2Vec/baomoi.model.bin', binary=True)
        self.data = pd.read_csv(filePath, encoding="utf-8")

    def compare_sentences(self, wordList1, wordList2):
        vectors1 = [self.model[word] for word in wordList1 if word in self.model]
        vectors2 = [self.model[word] for word in wordList2 if word in self.model]

        # Average the word vectors for each sentence
        vector1 = np.mean(vectors1, axis=0)
        vector2 = np.mean(vectors2, axis=0)

        cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return cosine

    def remove_noiseData(self):
        result = {}
        for label, features in self.data.values:
            result[label] = result.get(label, [])
            result[label].append(preprocess_text(features))

        for label, wordList in result.items():
            result_ = {(i, j): self.compare_sentences(wordList[i], wordList[j])
                       for i in range(0, len(wordList) - 1) for j in range(i+1, len(wordList))}
            deletedIndex = set()
            for pair, cosine in result_.items():
                if cosine > 0.9:
                    deletedIndex.add(pair[0])

            wordList = np.array(wordList, dtype=object)
            wordList = np.delete(wordList, list(deletedIndex), axis=0)
            result[label] = list(wordList)
        return result

    def writeNewFile(self, result):
        newData = []
        for label, wordList in result.items():
            for sentence in wordList:
                newData.append([label, " ".join(sentence)])

        df = pd.DataFrame(newData, columns=["Nhãn", "Features"])
        df.to_csv(f + "/datasets/additionalData.csv", encoding="utf-8", index=False)

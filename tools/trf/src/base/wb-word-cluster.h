// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2014-2015 Tsinghua University
// Author: wb.th08@gmail.com (Bin Wang), ozj@tsinghua.edu.cn (Zhijian Ou)
//
// All h, cpp, cc, and script files (e.g. bat, sh, pl, py) should include the above
// license declaration. Different coding language may use different comment styles.

#ifndef _WB_WORD_CLUSTER_H_
#define _WB_WORD_CLUSTER_H_

#include "wb-system.h"

namespace wb
{
    /**
    @defgroup cluster Word Clustering
    @{
    */
	/**
	* \class
	* \brief word聚类
	*/
	class WordCluster
	{
	public:
		LHash<int, int> m_wordCount;  ///< N(w)
		LHash<int, int> m_classCount; ///< N(g)
		Trie<int, int> m_wordGramCount;  ///< N(w,v)
		Trie<int, int> m_invWordGram; ///< 储存每个w的前继，不计数，仅用于索引每个w的前继v
		//wbTrie<int, int> m_classGramCount; ///< N(g_w,g_v);
		int **m_pClassGramCount; ///< N(g_w,g_v);
		Trie<int, int> m_wordClassCount; ///< N(w,g), 储存时，w在前，g在后
		Trie<int, int> m_classWordCount; ///< N(g,w), 储存时，w在前，g在后

		double m_dWordLogSum; ///< 记录sum{N(w)logN(w)} ,因为仅仅需要计算一次

		Array<int> m_aClass; ///< 记录每个词w所在的类g
		int m_nClassNum;
		int m_nVocabSize; ///< word-id的个数
		int m_nSentNum; ///< 文本中的词总数

		int m_nUnigramNum;
		int m_nBigramNum;

		char *m_pathWordClass;
		char *m_pathClassWord;
		char *m_pathTagVocab;


	public:
		WordCluster(int nClass) : m_nClassNum(nClass){
			SAFE_NEW_DARRAY(m_pClassGramCount, int, nClass + 1, nClass + 1);

			m_pathWordClass = NULL;
			m_pathClassWord = NULL;
			m_pathTagVocab = NULL;
		};
		~WordCluster(void) {
			SAFE_DELETE_DARRAY(m_pClassGramCount, m_nClassNum + 1);
		};

		void Reverse(int *pGram) { int n = pGram[0]; pGram[0] = pGram[1]; pGram[1] = n; }
		void InitCount(const char *path, const char *pTagVocab = NULL);
		void UpdataCount();
		void CountAdd(LHash<int, int> &count, int nWord, int nAdd) {
			bool bFound;
			int *pCount = count.Insert(nWord, bFound);
			if (!bFound) *pCount = nAdd;
			else *pCount += nAdd;
		}
		void CountAdd(Trie<int, int> &count, int *pWord, int nLen, int nAdd) {
			bool bFound;
			int *pCount = count.Insert(pWord, nLen, bFound);
			if (!bFound) *pCount = nAdd;
			else *pCount += nAdd;
		}
		void CountAdd(int **pCount, int *pWord, int nLen, int nAdd) {
			pCount[pWord[0]][pWord[1]] += nAdd;
		}
		void WriteCount(LHash<int, int> &count, File &file);
		void WriteCount(Trie<int, int> &count, File &file, bool bReverse = false);
		void WriteRes_WordClass(const char *path);
		void WriteRes_ClassWord(const char *path);
		void WriteRes_TagVocab(const char *path);
		void Read_TagVocab(const char *path);

		double LogLikelihood();
		void MoveWord(int nWord, bool bOut = true);
		/// exchange the nWord form m_aClass[nWord] to nToClass
		void ExchangeWord(int nWord, int nToClass);

		void Cluster(int nMaxTime = -1);

		///使用出现频率进行简单的分类，不需要迭代
		void SimpleCluster();
	};

	/**
	 * \class
	 * \brief word cluster for multiple threads. w,v denote the words, g denote the corresponding class
	 */
	class WordCluster_t
	{
	public:
		LHash<int, int> m_word_count;		///< N(w) index the word unigram count
		Trie<int, int> m_wgram_count;   ///< N(w,v) word bigram count
		Trie<int, int> m_inv_wgram_count;		///< N(v,w) inverse word bigram count

		LHash<int, int> m_class;			///< index the class
		Trie<int, int> m_class_gram;		///< (g(w), g(v)) the index of the class ngram
		Trie<int, int> m_word_class_gram;	///< (w,g) the word-class ngram
		Trie<int, int> m_class_word_gram;	///< (g,w) the class-word ngram

		Mat<int> m_tCountBuf;		///< the count buffer for each threads
		Mat<int> m_tMap;			///< map the word to correspond class at each thread

		Array<int> m_mCountBuf;		///< the count buffer in main threads
		Array<int> m_mMap;			///< the final g(w)


		int m_nClassNum;	///< the maximum class number
		int m_nVocabSize;	///< word number, i.e. the maximum word-id + 1
		int m_nSentNum;		///< total sentence number

		int m_nUnigramNum;
		int m_nBigramNum;
		double m_dWordLogSum;

		String m_pathRes;		///< the result file,   [ w   g(w) ]

// 		char *m_pathWordClass;
// 		char *m_pathClassWord;
// 		char *m_pathTagVocab;

		//WordCluster cluster;

	public:
		WordCluster_t(int nClass, char *pathRes = NULL):
			//cluster(nClass),
			m_nClassNum(nClass){
// 			m_pathWordClass = NULL;
// 			m_pathClassWord = NULL;
// 			m_pathTagVocab = NULL;
			if (pathRes == NULL) {
				m_pathRes = "word_cluster.default.res";
			}
			else {
				m_pathRes = pathRes;
			}
		};
		~WordCluster_t(void) {
		};
		void WriteRes(const char *path);
		void ReadRes(const char *path);
		void Reverse(int *pGram) { int n = pGram[0]; pGram[0] = pGram[1]; pGram[1] = n; }
		void InitCount(const char *path, const char *path_init_res = NULL);
		void UpdateCount(Array<int> &aCountBuf);
		void CountAdd(Array<int> &aCountBuf, LHash<int, int> &hash, int key, int count);
		void CountAdd(Array<int> &aCountBuf, Trie<int, int> &hash, int *pKey, int nLen, int count);
		void CountAdd(VecShell<int> &aCountBuf, LHash<int, int> &hash, int key, int count);
		void CountAdd(VecShell<int> &aCountBuf, Trie<int, int> &hash, int *pKey, int nLen, int count);

		void CopyCountToThreads(Array<int> &aCountBuf);
		/// move word in/out of a class and update the counts
		void MoveWord(VecShell<int> vCountBuf, VecShell<int> vMap, int nWord, bool bOut = true);
		/// exchange the nWord form m_aClass[nWord] to nToClass
		void ExchangeWord(VecShell<int> vCountBuf, VecShell<int> vMap, int nWord, int nToClass);
		/// cluster
		void Cluster(int nMaxTime = -1);
		/// claculate the Loglikelihood
		double LogLikelihood(VecShell<int> vCountBuf);

		///使用出现频率进行简单的分类，不需要迭代
		void SimpleCluster();
	};
	/** @} */
}

#endif

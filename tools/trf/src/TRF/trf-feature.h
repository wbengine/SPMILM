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


#pragma once
#include "trf-vocab.h"
#include <omp.h>

namespace trf
{
	const int c_nMaxOrder = 100;

	class Seq;
	class FeatStyle;
	class FeatTable;
	class Feat;

#define word_layer 0
#define class_layer 1
    /**
    @defgroup feature Feature
    this module defines the classes relative to the features.
    @{
    */
	/**
	 * \brief define a sequence including the word sequence and class sequence
	 */
	class Seq
	{
	public:
		Mat<VocabID> x;
		int nLen;
		int nMaxLen; ///< to denote the buffer size
	public:
		Seq() :nLen(0),nMaxLen(0) {}
		Seq(int len) :nLen(0),nMaxLen(0) { Reset(len); }
		/// reset only change the len variable, does not change the buffer size.
		void Reset(int p_len) {
			if (p_len > nMaxLen) {
				Mat<VocabID> newx(2, p_len);
				newx.Fill(0);
				for (int i = 0; i < nMaxLen; i++) {
					newx[0][i] = x[0][i];
					newx[1][i] = x[1][i];
				}
				x.Copy(newx);
				nMaxLen = p_len;
			}

			nLen = p_len;
		}
		/// copy the sequence
		void Copy(Seq &seq) {
			x.Copy(seq.x);
			nLen = seq.nLen;
			nMaxLen = seq.nMaxLen;
		}
		int GetLen() const { return nLen; }
		/// transform the word sequence (form file) to Seq
		void Set(Array<int> &aInt, Vocab *pv);
		void Set(int *pInt, int nLen, Vocab *pv);
		/// Random
		void Random(Vocab *pv);
		/// set the class based the word sequence
		void SetClass(Vocab *pv);
		/// get word sequence
		VocabID *GetWordSeq() { return x[word_layer].GetBuf(); }
		/// get class sequence
		VocabID* GetClassSeq() { return x[class_layer].GetBuf(); }
		void Print();
		void Print(File &file);
	};

	/**
	 * \class FeatStyle
	 * \brief Analyse a determinate feat style (without ":")
	 */
	class FeatStyle
	{
	public:
		typedef struct {
			char c;  ///< the charactor "w" or "c"
			short i; ///< if "w" then i=0, if "c" then i=1, used to index the value in Seq
			short n; ///< number, "w[2]" then n=2
		} Field;
	public:
		int m_nOrder; ///< the total order of this style, including the skip distance
		Array<Field> m_aFields; ///< each field
	public:
		FeatStyle() :m_nOrder(0) {}
		FeatStyle(const char *pstyle) { Set(pstyle); }
		/// set and analyze the style
		void Set(const char *pstyle);
		/// map a ngram to the index key, return if get a correct key
		bool GetKey(Seq &seq, int nPos, int nOrder, Array<int> &key);
	};
	/**
	* \class FeatTable
	* \author wangbin
	* \date 2015-12-10
	* \brief define the feature style. such as "w3"(word-3gram); "c2"(class-2gram);
	*
	* using "w" denotes the word, "c" denote the class, "-" denote the skip, and "0-9" in "[]" denote the order
	* different orders correspond to different template
	* support the skip-gram, such as :
	*	-# "w[3]" (word-3gram); "c[2]" (class-2gram); "w[1:5]" (word ngram with order 1 to 5)
	*	-# "w[2]-[2]w[1]" (skip-word-3gram, ww--w);
	*   -# "w[2]-[1]c[2]" (skip-word-class-gram, ww--cc);
	*	-# "w[2]-[1:2]w[1]" (tried-skip-word-gram, ww [1:2] w);
	*/
	class FeatTable
	{
	public:
		String m_style; ///< using a string array to store the feature styles in this table, such as w2, w3,...
		int m_nMinOrder; ///< the ngram minimum order, including the skip distance
		int m_nMaxOrder; ///< the ngram maximum order, including the skip distance
		int m_nNum; ///< the ngram number
		Array<int> m_aCutoff; ///< cutoff setting for different order
		Trie<VocabID, int> *m_ptrie; ///< index all the features.

		Array<FeatStyle*> m_aStyleInfo;

		Array<Array<int> *> m_aKey; ///< define the key for each thread

	public:
		/// constructor
		FeatTable(const char *pstyle = "");
		/// destructor
		~FeatTable();
		/// Reset
		void Reset(const char *pstyle);
		/// get number
		int GetNum() const { return m_nNum; }
		/// get minimum order
		int GetMinOrder() const { return m_nMinOrder; }
		/// get maxmum order
		int GetMaxOrder() const { return m_nMaxOrder; }
		/// get style string
		const char *GetStyle() const { return m_style.GetBuffer(); }
		/// Extract a feature from a sequence
		void LoadFeat(Seq &seq);
		/// Set the number of each features. We should index the feature in defferent tables.
		void IndexFeat(int begIdx);
		/// cutoff the features
		int CutoffFeat();
		/// Find the corresponding feature using a key. This will return the beg/end ngram
		void Find(Array<int> &afeat, Array<int> &key, bool bBeg, bool bEnd);
		/// Find the ngram feature with a fixed order.
		void Find(Array<int> &afeat, Seq &seq, int pos, int order);
		/// Find all the feature in the sequence
		void Find(Array<int> &afeat, Seq &seq);
		/// Find all the feature depending on position
		void FindPosDep(Array<int> &afeat, Seq &seq, int pos);

		/// Read form file
		void ReadT(File &file, PValue *pValue = NULL);
		/// Write to file
		void WriteT(File &file, PValue *pValue = NULL);
// 		/// Read from binary
// 		void ReadB(File &file);
// 		/// Write to binary
// 		void WriteB(File &file);
	};

	/// include all the feature table
	class Feat
	{
	public:
		Array<FeatTable*> m_aTable; ///< different feature table
		int m_nTotalNum; ///< total feature number
	public:
		Feat(int nOrder = 0, bool bClass = true)
		{
			m_nTotalNum = 0;
			if (nOrder > 0) {
				Reset(nOrder, bClass);
			}
		}
		~Feat() { SAFE_DEL_POINTER_ARRAY(m_aTable); }
		/// Reset, set the order. Node: the maximum order (including the skip) may be larger than nOrder.
		void Reset(int nOrder, bool bClass);
		/// Reset, read a feature type files
		void Reset(const char *pfeatType);
		/// Get maximum order
		int GetMaxOrder();
		/// Get number
		int GetNum() const { return m_nTotalNum;  }
		/// Find the ngram feature with a fixed order.
		void Find(Array<int> &afeat, Seq &seq, int pos, int order);
		/// Find all the feature in the sequence
		void Find(Array<int> &afeat, Seq &seq);
		/// Find the class ngram feature with a fixed order
		void FindClass(Array<int> &afeat, Seq &seq, int pos, int order);
		/// Find the ngram feature depending on word[pos]
		void FindWord(Array<int> &afeat, Seq &seq, int pos, int order);
		/// Find the class ngram depending on the nPos
		void FindPosDep(Array<int> &afeat, Seq &seq, int pos, int type = 0);
		/// Load Features from corpus
		void LoadFeatFromCorpus(const char *path, Vocab *pv);
		/// Write the features
		void WriteT(File &file, PValue *pValue = NULL);
		/// Read the features
		void ReadT(File &file, PValue *pValue = NULL);
	};
	/** @} */
}

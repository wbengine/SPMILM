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
// All h, cpp, cc, cu, cuh and script files (e.g. bat, sh, pl, py) should include the above 
// license declaration. Different coding language may use different comment styles.

#ifndef _CUTRF_FEATURE_CUH_
#define _CUTRF_FEATURE_CUH_
#include "cutrf-vocab.cuh"
#include "trf-feature.h"

namespace cutrf
{
	/**
	* \brief define a sequence including the word sequence and class sequence
	*/
	class Seq
	{
	public:
		cu::Mat<trf::VocabID> x;
		int nLen;
		int nMaxLen; ///< to denote the buffer size
	public:
		__host__ Seq() :nLen(0), nMaxLen(0) {}
		__host__ Seq(trf::Seq &seq) { Copy(seq); }
		__host__ void Release() { x.Reset(); }
		__host__ void Init(int maxlen)
		{
			x.Reset(2, maxlen);
			nMaxLen = maxlen;
			nLen = 0;
		}
		__host__ void Copy(trf::Seq &seq) {
			if (nMaxLen < seq.nMaxLen) {
				Init(seq.nMaxLen);
			}
			nLen = seq.GetLen();
			seq.Reset(nMaxLen); // make the 2D memory space is total the same with current space.
			x.Copy(seq.x);
			seq.Reset(nLen); // back to the original length
		}
		__host__ void CopyTo(trf::Seq &seq) {
			x.CopyTo(seq.x);
			seq.nLen = nLen;
			seq.nMaxLen = nMaxLen;
		}
		/// reset only change the len variable, does not change the buffer size.
		__host__ __device__ void Reset(int p_len) {
#ifdef _DEBUG
			if (p_len >= nMaxLen) {
				printf("[ERROR] [Seq] Reset: the length(%d) >= max-len(%d)!!!", p_len, nMaxLen);
			}
#endif
			
			nLen = p_len;
		}
		__host__ __device__ int GetLen() const { return nLen; }
		/// set the class based the word sequence
		__device__ void SetClass(Vocab *v) {
			v->GetClass(x[class_layer].GetBuf(), x[word_layer].GetBuf(), nLen);
		}
		/// get word sequence
		__device__ VocabID *GetWordSeq() { return x[word_layer].GetBuf(); }
		/// get class sequence
		__device__ VocabID *GetClassSeq() { return x[class_layer].GetBuf(); }
	};

	/**
	* \class FeatStyle
	* \brief Analyse a determinate feat style (without ":")
	*/
	class FeatStyle
	{
	public:
		int m_nOrder; ///< the total order of this style, including the skip distance
		cu::Vec<trf::FeatStyle::Field> m_aFields; ///< each field
	public:
		__host__ FeatStyle() :m_nOrder(0) {}
		__host__ void Release() {
			m_aFields.Reset();
			m_nOrder = 0;
		}
		__host__ void Copy(trf::FeatStyle &style) {
			m_nOrder = style.m_nOrder;
			m_aFields.Copy(style.m_aFields);
		}
		/// map a ngram to the index key, return if get a correct key
		__device__ bool GetKey(Seq &seq, int nPos, int nOrder, cu::Array<int> &key)
		{
			key.Clean();
			if (m_nOrder > nOrder) {
				return false;
			}

			int nCur = nPos;
			for (int nfield = 0; nfield < m_aFields.GetSize(); nfield++) {
				trf::FeatStyle::Field info = m_aFields[nfield];
				for (int i = 0; i < info.n; i++) {
					if (info.i >= 0) {
						key.Add(seq.x[info.i][nCur]);	
					}
					nCur++;
				}
			}
			return true;
		}
	};

#define FEAT_KEY_MAXLEN 10
#define FEAT_KEY_BUF(p) p + (blockIdx.x * blockDim.x + threadIdx.x) * FEAT_KEY_MAXLEN

	/**
	 * \class FeatTable
	 * \class FeatTable on cuda
	 */
	class FeatTable
	{
	public:
		int m_nMinOrder; ///< the ngram minimum order, including the skip distance
		int m_nMaxOrder; ///< the ngram maximum order, including the skip distance
		int m_nNum; ///< the ngram number
		cu::Trie<VocabID, int> m_trie; ///< index all the features. 

		FeatStyle* m_aStyleInfo_host; ///< a vector on host used to release each FeatStyle
		cu::Vec<FeatStyle> m_aStyleInfo;  ///< a vector on device used in kernal function

		int *m_pKeyBuf;
	public:
		/// constructor
		__host__ FeatTable() : m_nMinOrder(0), m_nMaxOrder(0), m_nNum(0), m_pKeyBuf(NULL) {};
		__host__ FeatTable(trf::FeatTable &feat, int *pKeyBuf):m_pKeyBuf(pKeyBuf) { Copy(feat); }
		/// Release
		__host__ void Release() {
			m_trie.Release();
			for (int i = 0; i < m_aStyleInfo.GetSize(); i++)
				m_aStyleInfo_host[i].Release();
			delete [] m_aStyleInfo_host;
			m_aStyleInfo.Reset();
			//m_key.Reset();
		}
		/// Copy
		__host__ void Copy(trf::FeatTable &feat) {
			m_nMinOrder = feat.GetMinOrder();
			m_nMaxOrder = feat.GetMaxOrder();
			m_nNum = feat.GetNum();
			m_trie.Copy(*feat.m_ptrie);

			
			m_aStyleInfo_host = new FeatStyle[feat.m_aStyleInfo.GetNum()];
			for (int i = 0; i < feat.m_aStyleInfo.GetNum(); i++) {
				m_aStyleInfo_host[i].Copy(*feat.m_aStyleInfo[i]);
			}
			m_aStyleInfo.Copy(wb::VecShell<FeatStyle>(m_aStyleInfo_host, feat.m_aStyleInfo.GetNum()));

			//m_key.Reset(m_nMaxOrder + 1);
		}
		/// get number
		__host__ __device__ int GetNum() const { return m_nNum; }
		/// get minimum order
		__host__ __device__ int GetMinOrder() const { return m_nMinOrder; }
		/// get maxmum order
		__host__ __device__ int GetMaxOrder() const { return m_nMaxOrder; }
		/// Find the corresponding feature using a key. This will return the beg/end ngram
		__device__ void Find(cu::Array<int> &afeat, cu::Array<int> &key, bool bBeg, bool bEnd)
		{
// 			printf("key.len=%d =", m_key.GetNum());
// 			for (int i = 0; i < m_key.GetNum(); i++) {
// 				printf("%d_", m_key[i]);
// 			}
// 			printf("\n");

			int *pValue = m_trie.Find(key.GetBuf(), key.GetNum());
			
			if (pValue) {
				afeat.Add(*pValue);
//				printf("Value=%d\n", *pValue);
			}

			if (bBeg) { // at the begining
				int begid = trf::VocabID_seqbeg;
				int nsub = m_trie.FindTrie(&begid, 1);
				if (nsub != -1) {
					pValue = m_trie.Find(key.GetBuf(), key.GetNum(), nsub);
					if (pValue)
						afeat.Add(*pValue);
				}
			}

			if (bEnd) { // at the end
				int nsub = m_trie.FindTrie(key.GetBuf(), key.GetNum());
				int endid = trf::VocabID_seqend;
				if (nsub != -1) {
					pValue = m_trie.Find(&endid, 1, nsub);
					if (pValue)
						afeat.Add(*pValue);
				}
			}
		}
		/// Find the ngram feature with a fixed order. 
		__device__ void Find(cu::Array<int> &afeat, Seq &seq, int pos, int order)
		{
			cu::Array<int> key(FEAT_KEY_BUF(m_pKeyBuf), FEAT_KEY_MAXLEN);
			for (int i = 0; i < m_aStyleInfo.GetSize(); i++) {
				if (m_aStyleInfo[i].m_nOrder == order) {
					m_aStyleInfo[i].GetKey(seq, pos, order, key);
					Find(afeat, key, pos == 0, pos + order == seq.GetLen());
				}
			}
		}
		/// Find all the feature in the sequence
		__device__ void Find(cu::Array<int> &afeat, Seq &seq)
		{
			cu::Array<int> key(FEAT_KEY_BUF(m_pKeyBuf), FEAT_KEY_MAXLEN);
			for (int pos = 0; pos < seq.GetLen(); pos++)
			{
				for (int i = 0; i < m_aStyleInfo.GetSize(); i++)
				{
					if (false == m_aStyleInfo[i].GetKey(seq, pos, seq.GetLen() - pos, key))
						continue;

					Find(afeat, key, pos == 0, pos + m_aStyleInfo[i].m_nOrder == seq.GetLen());
				}
			}
		}
		/// Find all the feature depending on position
		__device__ void FindPosDep(cu::Array<int> &afeat, Seq &seq, int pos)
		{
			int nLen = seq.GetLen();
			cu::Array<int> key(FEAT_KEY_BUF(m_pKeyBuf), FEAT_KEY_MAXLEN);

			for (int i = 0; i < m_aStyleInfo.GetSize(); i++) {
				int order = m_aStyleInfo[i].m_nOrder;
				for (int n = max(0, pos - order + 1); n <= min(nLen - order, pos); n++) {
					bool bSucc = m_aStyleInfo[i].GetKey(seq, n, order, key);
					Find(afeat, key, n == 0, n + order == nLen);
				}
			}
		}
	};

	/// include all the feature table
	class Feat
	{
	public:
		FeatTable *m_aTable_host; ///< a host vector used to release each element.
		cu::Vec<FeatTable> m_aTable; ///< different feature table
		cu::Vec<char> m_aTableFlag; ///< using 'w' to denote containing word info, using 'c' denote containing only class
		int m_nTotalNum; ///< total feature number

		cu::Vec<int> m_KeyBuf; ///< the buffer for the key of each thread
	public:
		__host__ Feat() : m_nTotalNum(0) {}
		__host__ Feat(trf::Feat &feat, int nMaxThread) { Copy(feat, nMaxThread); }
		__host__ void Copy(trf::Feat &feat, int nMaxThread) {
			m_nTotalNum = feat.m_nTotalNum;

			m_KeyBuf.Reset(nMaxThread * FEAT_KEY_MAXLEN);

			m_aTable_host = new FeatTable[feat.m_aTable.GetNum()];
			for (int i = 0; i < feat.m_aTable.GetNum(); i++) {
				m_aTable_host[i].Copy(*feat.m_aTable[i]);
				m_aTable_host[i].m_pKeyBuf = m_KeyBuf.GetBuf();
			}
			m_aTable.Copy(wb::VecShell<FeatTable>(m_aTable_host, feat.m_aTable.GetNum()));

			/* flag */
			wb::Vec<char> vFlag(feat.m_aTable.GetNum());
			for (int i = 0; i < vFlag.GetSize(); i++) {
				if (NULL == strchr(feat.m_aTable[i]->GetStyle(), 'w')) {
					vFlag[i] = 'c';
				}
				else {
					vFlag[i] = 'w';
				}
			}
			m_aTableFlag.Copy(vFlag);
		}
		__host__ void Release()
		{
			for (int i = 0; i < m_aTable.GetSize(); i++) {
				m_aTable_host[i].Release();
			}
			delete[] m_aTable_host;
			m_aTable.Reset();
			m_aTableFlag.Reset();
		}
		/// Get maximum order
		__device__ int GetMaxOrder() {
			int maxOrder = 0;
			for (int i = 0; i < m_aTable.GetSize(); i++)
				maxOrder = max(m_aTable[i].m_nMaxOrder, maxOrder);
			return maxOrder;
		}
		/// Get number
		__host__ __device__ int GetNum() const { return m_nTotalNum; }
		/// Find the ngram feature with a fixed order. 
		__device__ void Find(cu::Array<int> &afeat, Seq &seq, int pos, int order)
		{
			for (int i = 0; i < m_aTable.GetSize(); i++) {
				m_aTable[i].Find(afeat, seq, pos, order);
			}
		}
		/// Find all the feature in the sequence
		__device__ void Find(cu::Array<int> &afeat, Seq &seq)
		{
			for (int i = 0; i < m_aTable.GetSize(); i++) {
				//printf("i=%d\n", i);
				m_aTable[i].Find(afeat, seq);
			}
		}
		/// Find the class ngram feature with a fixed order
		__device__ void FindClass(cu::Array<int> &afeat, Seq &seq, int pos, int order)
		{
			for (int i = 0; i < m_aTable.GetSize(); i++) {
				if (m_aTableFlag[i] == 'c') {
					/* no word, which means containing only class */
					m_aTable[i].Find(afeat, seq, pos, order);
				}
			}
		}
		/// Find the ngram feature depending on word[pos]
		__device__ void FindWord(cu::Array<int> &afeat, Seq &seq, int pos, int order)
		{
			for (int i = 0; i < m_aTable.GetSize(); i++) {
				if (m_aTableFlag[i] == 'w') {
					/* containing word */
					m_aTable[i].Find(afeat, seq, pos, order);
				}
			}
		}
	};
}

#endif
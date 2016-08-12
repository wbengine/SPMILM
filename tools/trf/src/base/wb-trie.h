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




/**
* \file
* \author WangBin
* \date 2016-04-29
* \brief Define the trie structure
*/


#ifndef _WB_TRIE_H_
#define _WB_TRIE_H_
#include "wb-lhash.h"

namespace wb
{

#define _wb_TRIE				wb::Trie<KeyT, DataT>
#define _wb_LHASH_FOR_TRIE		wb::LHash<KeyT, wb::Trie<KeyT, DataT>*>
#define _wb_LHASHITER_FOR_TRIE	wb::LHashIter<KeyT, wb::Trie<KeyT, DataT>*>
#define _wb_UNIT_FOR_TRIE		wb::LHash::Unit<KeyT, wb::Trie<KeyT, DataT>*>

	template <class KeyT, class DataT> class Trie;
	template <class KeyT, class DataT> class TrieIter;
	template <class KeyT, class DataT> class TrieIter2;


	/** 
	 * \brief trie structure
	 *
	 Trie��ʹ����һ����Ҫע������⣺
	 ��������һ����״�ṹ����ˣ�����߽׵�index�������ͽ׽ڵ㣬�磺����abc��������a��ab�ڵ㣬���ʹ��Find��Insert�����Żص�
	 bFoundΪtrue�����Ⲣ����ζ�����ǲ����a��ab�������ȷ���ж��Ƿ����ķ���Ϊ��\n
	 \code{.cpp}
		DataT *p = trie.Find(index, len);
		if ( !p ) {
		*trie.Insert(index, len) = %��һ��ֵ%
			%û�г���%
		} else {
			%�г���%
		}
	 \endcode
	 ��ΪFind�������жϵ�ǰ�ڵ㴦����ֵ�Ƿ񱻸���ֵ��������ֵ����Ż���ָ�룻���򣬷���NULL
	 *
	 */
	template <class KeyT, class DataT>
	class Trie
	{
		friend class TrieIter<KeyT, DataT>;
		friend class TrieIter2<KeyT, DataT>;
	private:
		DataT m_value; ///< the value 
		_wb_LHASH_FOR_TRIE *m_phash; ///< the hash containing the pointer to the sub-trie
	public:
		Trie() :m_phash(NULL){ Map_noKey(m_value); }
		~Trie() { Release();  }
		/// Release all the trie
		void Release()
		{
			if (m_phash)
			{
				KeyT key;
				_wb_TRIE **ppsub;
				_wb_LHASHITER_FOR_TRIE iter(m_phash);
				while (ppsub = iter.Next(key)) {
					(*ppsub)->Release();
				}
			}
			
			Map_noKey(m_value);
			if (m_phash) {
				delete m_phash;
				m_phash = NULL;
			}
		}
		/// Compute the total memory cost of the trie structure
		size_t TotalMemCost()
		{
			size_t nSize = sizeof(*this);
			if (m_phash) {
				KeyT key;
				nSize += m_phash->TotalMemCost();
				_wb_TRIE **ppsub;
				_wb_LHASHITER_FOR_TRIE iter(m_phash);
				while (ppsub = iter.Next(key)) {
					nSize += (*ppsub)->TotalMemCost();
				}
			}
			return nSize;
		}
		/// Clean
		/* For Trie, clean is equal to release */
		void Clean() { Release();  }
		/// set all the values to d
		void Fill(DataT d)
		{
			KeyT key;
			_wb_TRIE **ppsub;
			_wb_LHASHITER_FOR_TRIE iter(m_phash);
			while (ppsub = iter.Next(key)) {
				(*ppsub)->Fill(d);
			}
			m_value = d;
		}
		/// Get value
		DataT* GetData() { return &m_value; }
		/// Get hash pointer
		_wb_LHASH_FOR_TRIE *GetHash() const { return m_phash; }
		/// detect if current trie have legal value
		bool IsDataLegal() { return !Map_noKeyP(m_value); }
		/// Find a value
		DataT* Find(const KeyT *p_pIndex, int nIndexLen, bool &bFound)
		{
			_wb_TRIE *psub = FindTrie(p_pIndex, nIndexLen, bFound);
			if (psub && psub->IsDataLegal()) {
				return psub->GetData();
			}
			bFound = false;
			return NULL;
		}
		/// Insert a value
		DataT* Insert(const KeyT *p_pIndex, int nIndexLen, bool &bFound)
		{
			_wb_TRIE *psub = InsertTrie(p_pIndex, nIndexLen, bFound);
			return psub->GetData();
		}
		/// Find a sub-trie
		_wb_TRIE *FindTrie(const KeyT *p_pIndex, int nIndexLen, bool &bFound)
		{
			if (nIndexLen == 0) {
				bFound = true;
				return this;
			}

			if (!m_phash) {
				bFound = false;
				return NULL;
			}
			
			_wb_TRIE **ppsub = m_phash->Find(p_pIndex[0], bFound);
			if (!bFound || !ppsub) {
				bFound = false;
				return NULL;
			}

			return (*ppsub)->FindTrie(p_pIndex + 1, nIndexLen - 1, bFound);
		}
		/// Insert a sub-trie
		_wb_TRIE *InsertTrie(const KeyT *p_pIndex, int nIndexLen, bool &bFound)
		{
			if (nIndexLen == 0) {
				return this;
			}

			if (!m_phash) {
				m_phash = new _wb_LHASH_FOR_TRIE;
			}
			_wb_TRIE **ppsub = m_phash->Insert(p_pIndex[0], bFound);
			if (!bFound) {
				*ppsub = new _wb_TRIE;
			}
			return (*ppsub)->InsertTrie(p_pIndex + 1, nIndexLen - 1, bFound);
		}
		/// Find a value
		DataT* Find(const KeyT *p_pIndex, int nIndexLen) 
		{
			bool bFound;
			return Find(p_pIndex, nIndexLen, bFound);
		}
		/// Insert a value
		DataT* Insert(const KeyT *p_pIndex, int nIndexLen) 
		{
			bool bFound;
			return Insert(p_pIndex, nIndexLen, bFound);
		}
		/// Find a sub-trie
		_wb_TRIE *FindTrie(const KeyT *p_pIndex, int nIndexLen)
		{
			bool bFound;
			return FindTrie(p_pIndex, nIndexLen, bFound);
		}
		/// Insert a sub-trie
		_wb_TRIE *InsertTrie(const KeyT *p_pIndex, int nIndexLen)
		{
			bool bFound;
			return InsertTrie(p_pIndex, nIndexLen, bFound);
		}
	};


	/**
	* \brief iter all the sub-tries
	*/
	template <class KeyT, class DataT>
	class TrieIter
	{
		friend class TrieIter2<KeyT, DataT>;
	public:
		TrieIter(_wb_TRIE *ptrie, bool(*sort)(KeyT, KeyT) = 0)
			: m_Iter(ptrie->m_phash, sort) {};
		/// Initialization
		void Init() { m_Iter.Init(); };
		/// Get next sub-trie
		/* the returned trie must contain a legal value*/
		_wb_TRIE *Next(KeyT &key) {
			_wb_TRIE *p;
			while (p = Next_1(key)) {
				if (p->IsDataLegal())
					break;
			}
			return p;
		}

	private:
		/// Get next sub-trie
		/* the returned trie may not contain a legal value */
		_wb_TRIE *Next_1(KeyT &key) {
			_wb_TRIE **pp = m_Iter.Next(key);
			return (pp) ? *pp : NULL;
		};
	private:
		_wb_LHASHITER_FOR_TRIE m_Iter; /// The iter for sub-trie
	};


	/**
	* \brief Get all the values whose indexes are of a fixed length. The returned tries may not contain a legal values
	*/
	template <class KeyT, class DataT>
	class TrieIter2
	{
	public:
		TrieIter2(_wb_TRIE *ptrie, KeyT *pIndex, int level, bool(*sort)(KeyT, KeyT) = 0)
			: m_ptrie(ptrie), m_pIndex(pIndex), m_nLevel(level),
			m_iter(ptrie, sort), m_sort(sort){
			m_pSubIter2 = NULL;
		};
		/// Initialization
		void Init()
		{
			m_iter.Init();
			if (m_pSubIter2)
				delete m_pSubIter2;
			m_pSubIter2 = NULL;
		}
		/// Get next trie
		/* The returned trie may not contain a legal value*/
		_wb_TRIE *Next()
		{
			if (m_nLevel == 0) {
				return m_ptrie;
			}
			else if (m_nLevel == 1) {
				return m_iter.Next(*m_pIndex);
			}

			while (1)
			{
				if (m_pSubIter2 == NULL) {
					_wb_TRIE *pSub = m_iter.Next_1(*m_pIndex);
					if (pSub == NULL)
						return NULL;

					m_pSubIter2 = new TrieIter2<KeyT, DataT>(pSub, m_pIndex + 1, m_nLevel - 1, m_sort);
				}

				_wb_TRIE *pNext = m_pSubIter2->Next();
				if (pNext == NULL) {
					delete m_pSubIter2;
					m_pSubIter2 = NULL;
				}
				else {
					return pNext;
				}
			}

		}

	private:
		TrieIter<KeyT, DataT> m_iter;  ///< iter for the sub-trie
		TrieIter2<KeyT, DataT> *m_pSubIter2; ///< iter2 for the sub-iter2
		_wb_TRIE *m_ptrie; ///< the root trie
		KeyT *m_pIndex; ///< the index
		int m_nLevel; ///< the index length
		bool(*m_sort)(KeyT, KeyT); ///< sort function
	};

}

#endif

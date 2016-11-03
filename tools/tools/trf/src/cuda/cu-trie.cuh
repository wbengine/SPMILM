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

/**
* \file
* \author WangBin
* \date 2016-05-20
* \brief define a trie structure on GPU 
*/

#ifndef _CU_TRIE_CUH_
#define _CU_TRIE_CUH_
#include "cu-def.cuh"
#include "cu-lhash.cuh"
#include "wb-log.h"
#include "wb-vector.h"
#include "wb-trie.h"

namespace cu
{

#define _cu_TRIE				cu::Trie<KeyT, DataT>
#define _cu_LHASH_FOR_TRIE		cu::LHash<KeyT, int> ///< a hash used to find the trie-id in array m_pInfo


	/*
	 * \brief A trie structure with __device__ function.
	 * \details 
	 All the functions with memory allocation/Free are __host__;\n
	 All the functions with memory read/write are __device__;\n
	 The rests are both.
	*/
	template <class KeyT, class DataT>
	class Trie
	{
	public:
		typedef struct {
			DataT m_data;   ///< the data 
			int m_nBufBeg;  ///< the buffer begin position in the whole buffer memory
			int m_nMaxBits; ///< The bit number of the memory buffer. (2^m_nMaxBits is the memory size)
			int m_nUnitNum; ///< The unit number add to the hash.
		} TrieInfo;
	public:
		Pair<KeyT,int> *m_pUnit; ///< the buffer storing all the hashs
		TrieInfo *m_pInfo; ///< the information of all the hashs
		MemLocation m_memloc; ///< record the memory in host or device.
	public:
		__host__ Trie() : m_pUnit(NULL), m_pInfo(NULL), m_memloc(none) {}
		/// The destructor doesn't release the buffer
		__host__ ~Trie() {  }
		/// Release
		__host__ void Release()
		{
			switch (m_memloc) {
			case host:
				SAFE_DELETE_ARRAY(m_pUnit);
				SAFE_DELETE_ARRAY(m_pInfo);
				break;
			case device:
				CUDA_CALL(cudaFree(m_pUnit));
				CUDA_CALL(cudaFree(m_pInfo));
				break;
			}
		}
		/// copy a trie from host to device memory
		__host__ void Copy(_wb_TRIE &trie)
		{
			m_memloc = device;
			wb::Array<Pair<KeyT, int>> aUnits;
			wb::Array<TrieInfo> aInfos;
			ShrinkTrie(trie, aUnits, aInfos);

			CUDA_CALL(cudaMalloc(&m_pUnit, sizeof(aUnits[0]) * aUnits.GetNum()));
			CUDA_CALL(cudaMalloc(&m_pInfo, sizeof(aInfos[0]) * aInfos.GetNum()));
			CUDA_CALL(cudaMemcpy(m_pUnit, aUnits.GetBuffer(), sizeof(aUnits[0])*aUnits.GetNum(), cudaMemcpyHostToDevice));
			CUDA_CALL(cudaMemcpy(m_pInfo, aInfos.GetBuffer(), sizeof(aInfos[0])*aInfos.GetNum(), cudaMemcpyHostToDevice));
		}
		/// Create a cu::trie in host memory
		__host__ void CopyToHost(_wb_TRIE &trie)
		{
			m_memloc = host;
			wb::Array<Pair<KeyT,int>> aUnits;
			wb::Array<TrieInfo> aInfos;
			ShrinkTrie(trie, aUnits, aInfos);

			m_pUnit = new Pair<KeyT, int>[aUnits.GetNum()];
			m_pInfo = new TrieInfo[aInfos.GetNum()];
			memcpy(m_pUnit, aUnits.GetBuffer(), aUnits.GetNum()*sizeof(m_pUnit[0]));
			memcpy(m_pInfo, aInfos.GetBuffer(), aInfos.GetNum()*sizeof(m_pInfo[0]));
		}
		/// map a trie sturce to a linear buffer style
		__host__ void ShrinkTrie(_wb_TRIE &trie, wb::Array<Pair<KeyT, int>> &aUnits, wb::Array<TrieInfo> &aInfos)
		{
			wb::LHash<void*, int> aMapTrie2Info;
			wb::Stack<_wb_TRIE*> stack;
			_wb_TRIE *pTrie;

			/* create the hash info */
			stack.Push(&trie);
			while (stack.Pop(&pTrie)) {
				_wb_LHASH_FOR_TRIE *pHash = pTrie->GetHash();
				// create a hash info / a trie node
				aInfos.Add();
				aInfos.End().m_data = *pTrie->GetData();
				aInfos.End().m_nBufBeg = 0; // set zero
				if (pHash) {
					aInfos.End().m_nMaxBits = pHash->GetMaxBits();
					aInfos.End().m_nUnitNum = pHash->GetNum();
				}
				else {
					aInfos.End().m_nMaxBits = 0;
					aInfos.End().m_nUnitNum = 0;
				}

				// save the mapping
				bool bFound;
				int *ppInfoNum = aMapTrie2Info.Insert(pTrie, bFound);
				if (bFound) {
					lout_error("Find a exist trie pointor!");
				}
				*ppInfoNum = aInfos.GetNum() - 1;

				// push the sub trie into the stack
				if (pHash) {
					KeyT key;
					_wb_TRIE **ppsub;
					_wb_LHASHITER_FOR_TRIE iter(pHash);
					while (ppsub = iter.Next(key)) {
						stack.Push(*ppsub);
					}
				}
			}

			/* create the buffer */
			stack.Clean();
			stack.Push(&trie);
			while (stack.Pop(&pTrie)) {
				_wb_LHASH_FOR_TRIE *pHash = pTrie->GetHash();

				// set the info
				int *pid = aMapTrie2Info.Find(pTrie);
				if (!pid) {
					lout_error("Cann't find the trie-id in the map");
				}
				aInfos[*pid].m_nBufBeg = aUnits.GetNum();

				// create the hash buffer
				if (pHash) {
					int nSize = pHash->GetSize();
					aUnits.SetNum(aUnits.GetNum() + nSize);
					auto *pNext = aUnits.GetBuffer(aUnits.GetNum() - nSize);
					auto pHashBuf = pHash->GetBuffer();
					for (int i = 0; i < nSize; i++) {
						pNext[i].key = pHashBuf[i].key;
						pNext[i].value = -1;
						if (!Map_noKeyP(pHashBuf[i].key)) {
							int *psub = aMapTrie2Info.Find(pHashBuf[i].value);
							if (!psub) {
								lout_error("Cann't find the sub-id in the map");
							}
							pNext[i].value = *psub;

							// push the sub trie into the stack
							stack.Push(pHashBuf[i].value);
						}
					}
				}
			}
		}
		/// create a cu::LHash of a trie
		__host__ __device__ _cu_LHASH_FOR_TRIE GetLHash(int nTrieId)
		{
			return _cu_LHASH_FOR_TRIE(m_pUnit + m_pInfo[nTrieId].m_nBufBeg, 
				m_pInfo[nTrieId].m_nMaxBits, 
				m_pInfo[nTrieId].m_nUnitNum);
		}
		/// find trie, return the trie id in array m_pInfo
		__host__ __device__ int FindTrie(const KeyT *p_pIndex, int nIndexLen, int nSubTrie = 0)
		{
			int nTrieId = nSubTrie;
			for (int i = 0; i < nIndexLen; i++) {
				_cu_LHASH_FOR_TRIE curHash = GetLHash(nTrieId);
				int *pSubId = curHash.Find(p_pIndex[i]);
				if (!pSubId) {
					nTrieId = -1;
					break;
				}
				nTrieId = *pSubId;
			}
			return nTrieId;
		}
		/// find value
		__host__ __device__ DataT *Find(const KeyT *p_pIndex, int nIndexLen, int nSubTrie = 0)
		{
			int nTrieId = FindTrie(p_pIndex, nIndexLen, nSubTrie);
			if (nTrieId == -1) {
				return NULL;
			}
			return &(m_pInfo[nTrieId].m_data);
		}
		
	};
}

#endif
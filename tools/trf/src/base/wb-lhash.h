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
* \brief linear hash, using 'open address' to handle collision
*/

#ifndef _WB_LHASH_H_
#define _WB_LHASH_H_
#include <iostream>
#include <cstring>
#include <cmath>
using namespace std;


#ifndef Bit2Size
#define Bit2Size(bits) ( (int)(1<<(bits)) )
#endif


namespace wb
{

	template <class KeyT, class DataT> class LHash;
	template <class KeyT, class DataT> class LHashIter;

	

	/// Copy a key
	template <class KeyT>
	inline KeyT Map_copyKey(KeyT key) { return key; }
	/// Copy a string key
	inline const char *Map_copyKey(const char *key) { return strdup(key); }

	
	/// Free a key.
	template <class KeyT>
	inline void Map_freeKey(KeyT key) {};
	/// Free a key, if the key is string
	inline void Map_freeKey(const char *key) { free((void*)key); }


	/** \name no-key values
	 *  - for signed type, no-key is the smallest negative value
	 *  - for unsigned type, no-key is the largest value
	 *  - for float/double type, no-key is a large float/double
	 */
	//@{
	const short c_ShortNokeyValue = (short)(1u << (sizeof(short) * 8 - 1)); ///< nokey value for short
	const int c_IntNokeyValue = (int)(1u << (sizeof(int) * 8 - 1)); ///< nokey value for int
	const long c_LongNokeyValue = (long)(1uL << (sizeof(long) * 8 - 1)); ///< nokey value for long
	const long long c_LLongNokeyValue = (long long)(1LL << (sizeof(long long) * 8 - 1)); ///< nokey value for longlong
	const short unsigned c_UShortNokeyValue = ~(short unsigned)0; ///< nokey value for unsigned short
	const unsigned c_UIntNokeyValue = ~(unsigned)0; ///< nokey value for unsigned int
	const long unsigned c_ULongNokeyValue = ~(long unsigned)0; ///< nokey value for unsigned long
	const float c_floatNokeyValue = 1e15; ///< nokey value for float
	const double c_doubleNokeyValue = 1e20; ///< nokey value for double
	//@}

	/** \name set key to no-key
	 */
	///@{
	template <class KeyT>
	inline void Map_noKey(KeyT *&key) { key = 0; }
	inline void Map_noKey(int &key) { key = c_IntNokeyValue; }
	inline void Map_noKey(short int &key) { key = c_ShortNokeyValue; }
	inline void Map_noKey(long int &key) { key = c_LongNokeyValue; }
	inline void Map_noKey(long long &key) { key = c_LLongNokeyValue; }
	inline void Map_noKey(unsigned &key) { key = c_UIntNokeyValue; }
	inline void Map_noKey(short unsigned &key) { key = c_UShortNokeyValue; }
	inline void Map_noKey(long unsigned &key) { key = c_ULongNokeyValue; }
	inline void Map_noKey(float &key) { key = c_floatNokeyValue; }
	inline void Map_noKey(double &key) { key = c_doubleNokeyValue; }
	///@}

	/** \name no-key detection
	 * detect if the key is no-key
	 */
	///@{
	template <class KeyT>
	inline bool Map_noKeyP(KeyT *key) { return key == 0; }
	inline bool Map_noKeyP(int key) { return key == c_IntNokeyValue; }
	inline bool Map_noKeyP(short int key) { return key == c_ShortNokeyValue; }
	inline bool Map_noKeyP(long int key) { return key == c_LongNokeyValue; }
	inline bool Map_noKeyP(long long key) { return key == c_LLongNokeyValue; }
	inline bool Map_noKeyP(unsigned key) { return key == c_UIntNokeyValue; }
	inline bool Map_noKeyP(short unsigned key) { return key == c_UShortNokeyValue; }
	inline bool Map_noKeyP(long unsigned key) { return key == c_ULongNokeyValue; }
	inline bool Map_noKeyP(float key) { return key == c_floatNokeyValue; }
	inline bool Map_noKeyP(double &key) { return key == c_doubleNokeyValue; }
	///@}

	/** \name Compare two Keys
	 * Return if the input keys are equal.
	 */
	///@{
	template <class KeyT>
	inline bool Map_equalKey(KeyT key1, KeyT key2) { return (key1 == key2); }
	inline bool Map_equalKey(float key1, float key2) { return fabs(key1 - key2) < 1e-2; }
	inline bool Map_equalKey(double key1, double key2) { return fabs(key1 - key2) < 1e-15; }
	inline bool Map_equalKey(const char *pStr1, const char *pStr2) { return strcmp(pStr1, pStr2) == 0; }
	///@}
	
#define HashMask(nbits) (~((~0L)<<(nbits)))

	/** @name HashKey
	 * Hashing functions.
	 * (We provide versions for integral types and char strings;
	 * user has to add more specialized definitions.)
	*/
	///@{
	/// hash for unsigned long
	inline unsigned long LHash_hashKey(unsigned long key, unsigned maxBits)
	{
		return (((key * 1103515245 + 12345) >> (30 - maxBits)) & HashMask(maxBits));
	}

	/// hash for pointer
	template <class KeyT>
	inline unsigned long LHash_hashKey(KeyT *key, unsigned maxBits)
	{
		return LHash_hashKey((unsigned long)key, maxBits);
	}

	/// hash for integral
	inline unsigned long LHash_hashKey(int key, unsigned maxBits)
	{
		return LHash_hashKey((unsigned long)key, maxBits);
	}

	/// hash for float
	inline unsigned long LHash_hashKey(float key, unsigned maxBits)
	{
		return LHash_hashKey((unsigned long)(100 * key), maxBits);
	}
	/// hash for double
	inline unsigned long LHash_hashKey(double key, unsigned maxBits)
	{
		return LHash_hashKey((unsigned long)(100 * key), maxBits);
	}
	/// hash for string
	/*
		* The string hash function was borrowed from the Tcl libary, by
		* John Ousterhout.  This is what he wrote:
		*
		* I tried a zillion different hash functions and asked many other
		* people for advice.  Many people had their own favorite functions,
		* all different, but no-one had much idea why they were good ones.
		* I chose the one below (multiply by 9 and add new character)
		* because of the following reasons:
		*
		* 1. Multiplying by 10 is perfect for keys that are decimal strings,
		*    and multiplying by 9 is just about as good.
		* 2. Times-9 is (shift-left-3) plus (old).  This means that each
		*    character's bits hang around in the low-order bits of the
		*    hash value for ever, plus they spread fairly rapidly up to
		*    the high-order bits to fill out the hash value.  This seems
		*    works well both for decimal and non-decimal strings.
	*/
	inline unsigned long LHash_hashKey(const char *key, unsigned maxBits)
	{
		unsigned long i = 0;

		for (; *key; key++) {
			i += (i << 3) + *key;
		}
		return LHash_hashKey(i, maxBits);
	}
	///@}

	/** @name SortFunction
	 * sort function for LHashIter
	 */
	///@{
	template <class KeyT>
	inline bool LHash_IncSort(KeyT k1, KeyT k2) { return k1 <= k2; }
	inline bool LHash_IncSort(const char *p1, const char *p2) { return strcmp(p2, p1) <= 0; }
	///@}

	const int cn_MinLHashBits = 3; ///< if the bits of hash less than this value, using linear table
	const int cn_MaxLHashBits = 31;	///< the maximum bits number support by hast
	const float cf_HashRatio = 0.8f; ///< fill-rate of the hash. 


	/**
	 * \brief a linear hash table
	 */
	template <class KeyT, class DataT>
	class LHash
	{
		friend class LHashIter<KeyT, DataT>;
	public:
		/// the Unit of hash
		typedef struct 
		{
			KeyT key; ///< the key 
			DataT value; ///< the value 
		} Unit;
	protected:
		int m_nMaxBits; ///< The bit number of the memory buffer. (2^m_nMaxBits is the memory size)
		int m_nUnitNum; ///< The unit number add to the hash.
		Unit *m_pUnit; ///< pointer to the buffer.

#ifdef _DEBUG
		static int nLocateNum;	  ///< search number
		static double fLocateStep;  ///< the average locate step
#endif

	public:
		/// constructor
		LHash(int nSize = 0) :m_nMaxBits(0), m_pUnit(NULL), m_nUnitNum(0) { Reset(nSize); }
		/// destructor
		~LHash() { Release(); }
		/// Get the unit number
		int GetNum() const { return m_nUnitNum; }
		/// Get the buffer size of hash
		int GetSize() const { return Bit2Size(m_nMaxBits); }
		/// Get the buffer size bits
		int GetMaxBits() const { return m_nMaxBits; }
		/// Get the buffer
		const Unit* GetBuffer() const { return m_pUnit; }
		/// Compute the whole memory cost of hash structure
		size_t TotalMemCost()
		{
			return sizeof(*this) + GetSize() * sizeof(Unit);
		}
		/// calculate the buffer size need to be allocated
		int RoundSize(int nSize)
		{
			if (nSize < Bit2Size(cn_MinLHashBits))
				return nSize;
			return (int)((nSize + 1) / cf_HashRatio);
		}
		/// Allocate the buffer
		void Alloc(int nSize)
		{
			int maxBits, maxUnits;

			//���㣬��Ҫ����nSize����Ҫ���ڴ�ռ��bits��
			maxBits = 1; //maxBits��СΪ1
			while (Bit2Size(maxBits) < nSize) {
				if (maxBits > cn_MaxLHashBits) {
					cout << "[LHash] Alloc: needed bits over cn_MaxLashBits" << endl;
					exit(1);
				}
				++maxBits;
			}

			maxUnits = Bit2Size(maxBits);
			if (m_pUnit) {
				delete[] m_pUnit;
				m_pUnit = NULL;
			}
			m_pUnit = new Unit[maxUnits];

			m_nMaxBits = maxBits;
			m_nUnitNum = 0;

			//�����е�ֵ������ΪnoKey
			for (int i = 0; i<maxUnits; i++)
				Map_noKey(m_pUnit[i].key);
		}
		/// Release the buffer
		void Release()
		{
			if (m_pUnit)
			{
				int maxUnits = Bit2Size(m_nMaxBits);
				for (int i = 0; i<maxUnits; i++){
					if (!Map_noKeyP(m_pUnit[i].key)) {
						Map_freeKey(m_pUnit[i].key);
					}
				}

				delete [] m_pUnit;
				m_pUnit = NULL;
			}
			m_nUnitNum = 0;
			m_nMaxBits = 0;
		}
		/// Reset the hash buffer
		void Reset(int nSize)
		{
			Release();
			if (nSize > 0)
				Alloc(RoundSize(nSize));
		}
		/// Clean the hash, but don't release the buffer
		void Clean()
		{
			if (m_pUnit) {
				int nMaxUnit = Bit2Size(m_nMaxBits);
				for (int i = 0; i<nMaxUnit; i++) {
					if (!Map_noKeyP(m_pUnit[i].key))
						Map_noKey(m_pUnit[i].key);
				}

				m_nUnitNum = 0;
			}
		}
		/// Set all the values to d
		void Fill(DataT d)
		{
			if (m_pUnit) {
				int nMaxUnit = Bit2Size(m_nMaxBits);
				for (int i = 0; i<nMaxUnit; i++) {
					if (!Map_noKeyP(m_pUnit[i].key))
						m_pUnit[i].value = d;
				}
			}
		}
		/// Find the key
		inline bool Locate(KeyT key, int &index) const
		{
			//lhash_assert(!Map_noKeyP(key));

			if (!m_pUnit)
				return false;


			if (m_nMaxBits < cn_MinLHashBits)
			{
				//���Բ���
				for (int i = 0; i<m_nUnitNum; i++) {
					if (Map_equalKey(key, m_pUnit[i].key)) {
						index = i;
						return true;
					}
				}

				index = m_nUnitNum;
				return false;
			}
			else
			{
#if _DEBUG
				int nStep = 1;
#endif
				//��ϣ����
				bool bFound = false;
				int nHash = LHash_hashKey(key, m_nMaxBits);
				for (int i = nHash;; i = (i + 1) % Bit2Size(m_nMaxBits)) {
					if (Map_noKeyP(m_pUnit[i].key)) {
						index = i;
						bFound = false;
						break;
					}
					else if (Map_equalKey(key, m_pUnit[i].key)) {
						index = i;
						bFound = true;
						break;
					}
#if _DEBUG
					nStep++;
#endif
				}

#if _DEBUG
				//��¼���д���
				LHash<KeyT, DataT>::fLocateStep =
					LHash<KeyT, DataT>::fLocateStep * LHash<KeyT, DataT>::nLocateNum / (LHash<KeyT, DataT>::nLocateNum + 1) +
					1.0 * nStep / (LHash<KeyT, DataT>::nLocateNum + 1);
				LHash<KeyT, DataT>::nLocateNum++;
#endif

				return bFound;
			}

			return false;
		}
		/// Find a value
		DataT *Find(KeyT key, bool &bFound)
		{
			int index;
			if ((bFound = Locate(key, index))) {
				return &(m_pUnit[index].value);
			}
			else {
				return NULL;
			}
		}
		/// Find a value
		DataT *Find(KeyT key) {
			bool bFound;
			return Find(key, bFound);
		}
		/// Insert a value
		DataT *Insert(KeyT key, bool &bFound)
		{
			int index;

			if (!m_pUnit)
				Alloc(1);

			if ((bFound = Locate(key, index))) {
				return &(m_pUnit[index].value);
			}
			else {
				int nNewSize = RoundSize(m_nUnitNum + 1);

				if (nNewSize > Bit2Size(m_nMaxBits)) { //��Ҫ���¿����ڴ�
					Unit *pOldUnit = m_pUnit;
					int nOldUnitNum = m_nUnitNum;
					int nOldMaxUnit = Bit2Size(m_nMaxBits);

					m_pUnit = NULL;  //��Ҫ���ָ�룬�������Alloc���ͷŵ��ڴ�
					Alloc(nNewSize); //�����ڴ�
					m_nUnitNum = nOldUnitNum;

					if (m_nMaxBits < cn_MinLHashBits) {
						//ֻ������copy��������
						memcpy(m_pUnit, pOldUnit, sizeof(pOldUnit[0])*nOldUnitNum);
					}
					else {
						// rehash
						for (int i = 0; i<nOldMaxUnit; i++) {
							KeyT curKey = pOldUnit[i].key;
							if (!Map_noKeyP(curKey)) {
								Locate(curKey, index);
								memcpy(&(m_pUnit[index]), &(pOldUnit[i]), sizeof(pOldUnit[i]));
							}
						}
					}

					delete[] pOldUnit;
					pOldUnit = NULL;

					//����Locate
					Locate(key, index);
				}

				m_pUnit[index].key = Map_copyKey(key);
				memset(&(m_pUnit[index].value), 0, sizeof(m_pUnit[index].value));
				new (&(m_pUnit[index].value)) DataT; //���ù��캯��

				m_nUnitNum++;
				return &(m_pUnit[index].value);
			}
		}
		/// Insert a value
		DataT *Insert(KeyT key)
		{
			bool bFound;
			return Insert(key, bFound);
		}
		/// Copy hash
		void Copy(LHash<KeyT, DataT> &other)
		{
			if (&other == this)
				return;

			if (other.m_pUnit) {
				int maxSize = Bit2Size(other.m_nMaxBits);
				Alloc(maxSize);
				for (int i = 0; i<maxSize; i++) {
					KeyT thisKey = other.m_pUnit[i].key;

					if (!Map_noKeyP(thisKey)) {
						m_pUnit[i].key = Map_copyKey(thisKey);

						new (&(m_pUnit[i].value)) DataT;

						m_pUnit[i].value = other.m_pUnit[i].value;
					}
				}
				m_nUnitNum = other.m_nUnitNum;
			}
			else {
				// 			if (m_pUnit)
				// 				cout<<"!!!!!"<<endl;
				Clean();
			}
		}
	};


#ifdef _DEBUG
	template <class KeyT, class DataT>
	int LHash<KeyT, DataT>::nLocateNum = 0;
	template <class KeyT, class DataT>
	double LHash<KeyT, DataT>::fLocateStep = 0;
#endif


	/**
	 * \brief the iter of LHash
	 */
	template <class KeyT, class DataT>
	class LHashIter
	{
	private:
		LHash<KeyT, DataT> *m_phash; ///< pointer to LHash
		int m_nCurIndex; ///< current index

		bool(*m_sortFun)(KeyT, KeyT); ///< sort function. If f(x1,x2)=true��then x1,x2 is the currect sort.
		KeyT *m_pSortedKey;   ///< save the sorted key
	public:
		/// constructor
		LHashIter(LHash<KeyT, DataT> *phash, bool(*sort)(KeyT, KeyT) = 0) : m_phash(phash),m_sortFun(sort)
		{
			m_nCurIndex = 0;
			m_pSortedKey = NULL;
			Init();
		}
		/// destructor
		~LHashIter()
		{
			if (m_pSortedKey)
				delete[] m_pSortedKey;
			m_pSortedKey = NULL;
		}
		/// initialize the iter
		void Init()
		{
			m_nCurIndex = 0;
			if (!m_phash)
				return;

			if (m_sortFun) {
				//��Key����
				if (m_pSortedKey)
					delete[] m_pSortedKey;
				m_pSortedKey = NULL;

				if (m_phash->GetNum() > 0) {
					m_pSortedKey = new KeyT[m_phash->GetNum()];
					int curKey = 0;

					int maxSize = Bit2Size(m_phash->m_nMaxBits);
					for (int i = 0; i<maxSize; i++) {
						if (!Map_noKeyP(m_phash->m_pUnit[i].key)) {
							m_pSortedKey[curKey] = m_phash->m_pUnit[i].key;
							curKey++;
						}
					}

					//����
					for (int i = 0; i<curKey - 1; i++) {
						for (int j = i + 1; j<curKey; j++) {
							if (!m_sortFun(m_pSortedKey[i], m_pSortedKey[j])) {
								KeyT t = m_pSortedKey[i];
								m_pSortedKey[i] = m_pSortedKey[j];
								m_pSortedKey[j] = t;
							}
						}
					}
				}

			}
		}
		/// get next value
		/**
		 * \param [out] key return the next key
		 * return return the pointer to the next value, to make it posible to modificate the value
		*/
		DataT* Next(KeyT &key)
		{
			if (!m_phash || m_phash->m_pUnit == NULL || m_phash->m_nUnitNum == 0)
				return NULL;

			if (m_sortFun)
			{
				//����
				if (m_nCurIndex == m_phash->GetNum())
					return NULL;

				key = m_pSortedKey[m_nCurIndex++];
				return m_phash->Find(key);
			}
			else
			{
				for (; m_nCurIndex < m_phash->GetSize(); m_nCurIndex++) {
					key = m_phash->m_pUnit[m_nCurIndex].key;
					if (!Map_noKeyP(key)) {
						return &(m_phash->m_pUnit[m_nCurIndex++].value);
					}
				}
			}

			return NULL;
		}
	};

	

}
#endif

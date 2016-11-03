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
* \date 2016-05-19
* \brief linear hash, using 'open address' to handle collision, used in GPU platform
*/

#ifndef _CU_LHASH_CUH_
#define _CU_LHASH_CUH_
#include "cu-def.cuh"
#include "cu-string.cuh"
#include "wb-lhash.h"

namespace cu
{
	/// Copy a key
	template <class KeyT>
	__host__ static KeyT Map_copyKey(KeyT key) { return key; }
	/// Copy a string key
	__host__ static char *Map_copyKey(const char *key) { return cu::strdup(key); }


	/// Free a key.
	template <class KeyT>
	__host__ static void Map_freeKey(KeyT key) {};
	/// Free a key, if the key is string
	__host__ static void Map_freeKey(char *key) { return cu::strfree(key); }


	/** \name no-key values
	*  - for signed type, no-key is the smallest negative value
	*  - for unsigned type, no-key is the largest value
	*  - for float/double type, no-key is a large float/double
	*/
	//@{
	__constant__ const short c_ShortNokeyValue = (short)(1u << (sizeof(short) * 8 - 1)); ///< nokey value for short
	__constant__ const int c_IntNokeyValue = (int)(1u << (sizeof(int) * 8 - 1)); ///< nokey value for int
	__constant__ const long c_LongNokeyValue = (long)(1uL << (sizeof(long) * 8 - 1)); ///< nokey value for log
	__constant__ const short unsigned c_UShortNokeyValue = ~(short unsigned)0; ///< nokey value for unsigned short
	__constant__ const unsigned c_UIntNokeyValue = ~(unsigned)0; ///< nokey value for unsigned int
	__constant__ const long unsigned c_ULongNokeyValue = ~(long unsigned)0; ///< nokey value for unsigned long
	__constant__ const float c_floatNokeyValue = 1e15; ///< nokey value for float
	__constant__ const double c_doubleNokeyValue = 1e20; ///< nokey value for double
	//@}

	/** \name set key to no-key
	*/
	///@{
	template <class KeyT>
	__host__ __device__ static void Map_noKey(KeyT *&key) { key = 0; }
	__host__ __device__ static void Map_noKey(int &key) { key = c_IntNokeyValue; }
	__host__ __device__ static void Map_noKey(short int &key) { key = c_ShortNokeyValue; }
	__host__ __device__ static void Map_noKey(long int &key) { key = c_LongNokeyValue; }
	__host__ __device__ static void Map_noKey(unsigned &key) { key = c_UIntNokeyValue; }
	__host__ __device__ static void Map_noKey(short unsigned &key) { key = c_UShortNokeyValue; }
	__host__ __device__ static void Map_noKey(long unsigned &key) { key = c_ULongNokeyValue; }
	__host__ __device__ static void Map_noKey(float &key) { key = c_floatNokeyValue; }
	__host__ __device__ static void Map_noKey(double &key) { key = c_doubleNokeyValue; }
	///@}

	/** \name no-key detection
	* detect if the key is no-key
	*/
	///@{
	template <class KeyT>
	__host__ __device__ static bool Map_noKeyP(KeyT *key) { return key == 0; }
	__host__ __device__ static bool Map_noKeyP(int key) { return key == c_IntNokeyValue; }
	__host__ __device__ static bool Map_noKeyP(short int key) { return key == c_ShortNokeyValue; }
	__host__ __device__ static bool Map_noKeyP(long int key) { return key == c_LongNokeyValue; }
	__host__ __device__ static bool Map_noKeyP(unsigned key) { return key == c_UIntNokeyValue; }
	__host__ __device__ static bool Map_noKeyP(short unsigned key) { return key == c_UShortNokeyValue; }
	__host__ __device__ static bool Map_noKeyP(long unsigned key) { return key == c_ULongNokeyValue; }
	__host__ __device__ static bool Map_noKeyP(float key) { return key == c_floatNokeyValue; }
	__host__ __device__ static bool Map_noKeyP(double &key) { return key == c_doubleNokeyValue; }
	///@}

	/** \name Compare two Keys
	* Return if the input keys are equal.
	*/
	///@{
	template <class KeyT>
	__host__ __device__ static bool Map_equalKey(KeyT key1, KeyT key2) { return (key1 == key2); }
	__host__ __device__ static bool Map_equalKey(float key1, float key2) { return fabs(key1 - key2) < 1e-2; }
	__host__ __device__ static bool Map_equalKey(double key1, double key2) { return fabs(key1 - key2) < 1e-15; }
	__host__ __device__ static bool Map_equalKey(const char *pStr1, const char *pStr2) { return cu::strcmp(pStr1, pStr2) == 0; }
	///@}

	template <class KeyT, class DataT>
	struct Pair{
		KeyT key; ///< the key
		DataT value; ///< the value
	};

	/*
	 * \class LHash
	 * \brief the Linear Hash class on GPU.
	 * \details the class only support the KeyT=integer, float/double or struct/class supporting "=" operator.
	 It doesn't support the KeyT=string, as handling string needs to allocate memory for each KeyT.
	 The destructor can't release the buffer and the buffer should be released in host code manually.
	*/
	template <class KeyT, class DataT>
	class LHash 
	{
	public:
		int m_nMaxBits; ///< The bit number of the memory buffer. (2^m_nMaxBits is the memory size)
		int m_nUnitNum; ///< The unit number add to the hash.
		Pair<KeyT,DataT> *m_pUnit; ///< pointer to the buffer.
	public:
		__host__ __device__ LHash() : m_nMaxBits(0),m_nUnitNum(0),m_pUnit(NULL) {}
		__host__ __device__ LHash(Pair<KeyT,DataT> *p, int nMaxBits, int nUnitNum) :
			m_pUnit(p), m_nMaxBits(nMaxBits), m_nUnitNum(nUnitNum) {}
		__host__ LHash(wb::LHash<KeyT,DataT> &h) { Copy(h); }
		/// !!! The destructor can't release the buffer and the buffer should be released in host code manually.
		__host__ __device__ ~LHash() { }
	public:
		/// Get the unit number
		__host__ __device__ int GetNum() const { return m_nUnitNum; }
		/// Get the buffer size of hash
		__host__ __device__ int GetSize() const { return Bit2Size(m_nMaxBits); }
		/// Get the buffer size bits
		__host__ __device__ int GetMaxBits() const { return m_nMaxBits; }
		/// Get the buffer
		__host__ __device__ const Pair<KeyT,DataT>* GetBuffer() const { return m_pUnit; }
		/// Compute the whole memory cost of hash structure
		__host__ size_t TotalMemCost()
		{
			return sizeof(*this) + GetSize() * sizeof(Pair<KeyT,DataT>);
		}
	
		/// Release the buffer
		__host__ void Release()
		{
			if (m_pUnit)
			{
				CUDA_CALL(cudaFree(m_pUnit));
				m_pUnit = NULL;
			}
			m_nUnitNum = 0;
			m_nMaxBits = 0;
		}
		/// copy a hash form host to device memory.
		/* This can not be used to copy pointer varibales */
		__host__ void Copy(wb::LHash<KeyT, DataT> &h)
		{
			m_nMaxBits = h.GetMaxBits();
			m_nUnitNum = h.GetNum();

			int nBytSize = h.GetSize() * sizeof(Pair<KeyT,DataT>);
			CUDA_CALL(cudaMalloc(&m_pUnit, nBytSize));
			CUDA_CALL(cudaMemcpy(m_pUnit, h.GetBuffer(), nBytSize, cudaMemcpyHostToDevice));
		}

		/// find a key, return true(success) and false(failed).
		__host__ __device__ bool Locate(KeyT key, int &index) const
		{
			if (!m_pUnit)
				return false;

			// if the array is small, then perform the linear search
			if (m_nMaxBits < cn_MinLHashBits)
			{
				for (int i = 0; i < m_nUnitNum; i++) {
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
				// hash search
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
				}

				return bFound;
			}

			return false;
		}
		/// Find a value
		__host__ __device__ DataT *Find(KeyT key, bool &bFound)
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
		__host__ __device__ DataT *Find(KeyT key) {
			bool bFound;
			return Find(key, bFound);
		}
		
	};

	/** @name HashKey
	* Hashing functions.
	* (We provide versions for integral types and char strings;
	* user has to add more specialized definitions.)
	*/
	///@{
	/// hash for unsigned long
	__host__ __device__ static unsigned long LHash_hashKey(unsigned long key, unsigned maxBits)
	{
		return (((key * 1103515245 + 12345) >> (30 - maxBits)) & HashMask(maxBits));
	}

	/// hash for pointer
	template <class KeyT>
	__host__ __device__ static unsigned long LHash_hashKey(KeyT *key, unsigned maxBits)
	{
		return LHash_hashKey((unsigned long)key, maxBits);
	}

	/// hash for integral
	__host__ __device__ static unsigned long LHash_hashKey(int key, unsigned maxBits)
	{
		return LHash_hashKey((unsigned long)key, maxBits);
	}

	/// hash for float
	__host__ __device__ static unsigned long LHash_hashKey(float key, unsigned maxBits)
	{
		return LHash_hashKey((unsigned long)(100 * key), maxBits);
	}
	/// hash for double
	__host__ __device__ static unsigned long LHash_hashKey(double key, unsigned maxBits)
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
// 	__host__ __device__ unsigned long cuLHash_hashKey(const char *key, unsigned maxBits)
// 	{
// 		unsigned long i = 0;
// 
// 		for (; *key; key++) {
// 			i += (i << 3) + *key;
// 		}
// 		return cuLHash_hashKey(i, maxBits);
// 	}
	///@}


}

#endif
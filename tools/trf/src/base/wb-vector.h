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
* \date 2013-9-2
* \brief Defination of simple dynamic array/stack/queue and so on.
*/



#ifndef _WB_VECTOR_H_
#define _WB_VECTOR_H_
//#include <conio.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdlib.h>
using namespace std;

//@{
/// memory allocation
#define POINT_TEST(p) { if ((p)==NULL) {cout<<#p<<"[New Memory Error]"<<endl; exit(0);} }
#define SAFE_NEW(p, Type) {p = new Type; POINT_TEST(p)}
#define SAFE_DNEW(PType, p, Type) { PType p = new Type; POINT_TEST(p) }
#define SAFE_NEW_ARRAY(p, Type, n) { p = new Type[n]; POINT_TEST(p) }
#define SAFE_NEW_DARRAY(p, Type, n, m) { p=new Type*[n]; for(int i=0; i<n; i++){p[i]=new Type[m];}; POINT_TEST(p) }
//@}

//@{
/// memory release
#define SAFE_DELETE(p) {if (p) delete (p); (p)=NULL;}
#define SAFE_DELETE_ARRAY(p) {if(p) delete [](p); (p)=NULL;}
#define SAFE_DELETE_DARRAY(p, n) { if(p){ for(int i=0; i<n; i++){delete [](p)[i];} delete [](p); (p)=NULL; } }
#define SAFE_DEL_POINTER_ARRAY(a) for (int i=0; i<a.GetNum(); i++) { SAFE_DELETE(a[i]); } a.Clean();
//@}

namespace wb
{
	/// defualt vector size
#define DEFAULE_VECTOR_SIZE 16
	/// transform the bit to memory size
#define VECTOR_bits2Size(n) (1<<n)

    /** \addtogroup struct
    @{
    */
	template <typename T>
	/**
	* \date 2016-04-28
	* \author Wangbin
	* \brief this is the basic class of Array/Stack/Queue. Realize the dynamic memory management
	*
	* If the size is full, then we re-new a new memory with double current memory size.
	*/
	class Vector
	{
	protected:
		T *m_pBuffer; ///< buffer pointer
		int m_nBits;  ///< bit number of memory size
	public:
		/// constructor, do not init the memeory to zero.
		Vector(int size = DEFAULE_VECTOR_SIZE)
		{
			m_pBuffer = NULL;
			m_nBits = Size2Bits(size);
			m_pBuffer = new T[VECTOR_bits2Size(m_nBits)];
		}
		/// desturctor
		~Vector()
		{
			if (m_pBuffer != NULL)
				delete[]m_pBuffer;
			m_pBuffer = NULL;
			m_nBits = 0;
		}
		/// get the memory size
		int Size() const { return VECTOR_bits2Size(m_nBits); }
		/// get the buffer pointer
		inline T* GetBuffer(int i = 0) const { return m_pBuffer + i; }
		/// get the value at position i
		inline T& Get(int i)
		{

			if (i<0) {
				i = 0;
			}

			int nSize = VECTOR_bits2Size(m_nBits);

			if (i >= nSize)
			{
				int nNewBits = Size2Bits(i + 1);

				T *pNew = new T[VECTOR_bits2Size(nNewBits)];
				//memset(pNew+m_nBufferSize, 0, sizeof(T)*nStepNum*m_nBufferStep);


				for (int n = 0; n<nSize; n++)
					pNew[n] = m_pBuffer[n];

				delete[] m_pBuffer;
				m_pBuffer = pNew;
				m_nBits = nNewBits;
			}

			return m_pBuffer[i];
		}

		/// Transform the size to bit
		inline int Size2Bits(int size)
		{
			for (int n = 1;; ++n) {
				if (VECTOR_bits2Size(n) >= size)
					return n;
			}
			return 0;
		}

		/// set all the values to m
		/** using '=' to set values. This may avoid the error caused by 'memcopy' */
		void Fill(T m)
		{
			for (int i = 0; i<VECTOR_bits2Size(m_nBits); i++)
				m_pBuffer[i] = m;
		}
		/// Fill all the memory to ZERO. call 'memset'
		void Zero()
		{
			memset(m_pBuffer, 0, sizeof(T)*VECTOR_bits2Size(m_nBits));
		}

		/// Copy a vector to another, using '=' to set values
		/*
			As sometimes using memcpy may cause error.
			Such as copying a string vector.
		*/
		void Copy(const Vector<T> &vector)
		{
			if (m_pBuffer == NULL) {
				m_nBits = vector.m_nBits;
				m_pBuffer = new T[VECTOR_bits2Size(m_nBits)];
			}
			else if (m_nBits < vector.m_nBits) {
				delete[] m_pBuffer;
				m_nBits = vector.m_nBits;
				m_pBuffer = new T[VECTOR_bits2Size(m_nBits)];
			}

			for (int i = 0; i<VECTOR_bits2Size(vector.m_nBits); i++)
				m_pBuffer[i] = vector.m_pBuffer[i];
		}
		/// Copy a vector to another, using 'memcpy' to set values
		void MemCopy(Vector<T> &vector)
		{
			if (m_pBuffer == NULL) {
				m_nBits = vector.m_nBits;
				m_pBuffer = new T[VECTOR_bits2Size(m_nBits)];
			}
			else if (m_nBits < vector.m_nBits) {
				delete[] m_pBuffer;
				m_nBits = vector.m_nBits;
				m_pBuffer = new T[VECTOR_bits2Size(m_nBits)];
			}

			memcpy(m_pBuffer, vector.m_pBuffer, sizeof(T)*VECTOR_bits2Size(m_nBits));
		}
		/// Set the buffer pointer to p_pt, and re-init the vector
		void BufferOutside(T* &p_pt)
		{
			p_pt = m_pBuffer;
			m_pBuffer = NULL;
			new (this) Vector();
		}
	};

	/**
	 * \class Array
	 * \date 2016-04-28
	 * \author WangBin
	 * \brief Dynamic array
	 * \note
		Set a[i] = a[j] is not safe as the memory is allocated dynamicly. Please use:
		temp = a[j];
		a[i] = temp;
	 */
	template <typename T>
	class Array : public Vector<T>
	{
	public:
		int m_nTop; ///< Record the top of the array.
	public:
		/// constructor
		Array(int size = DEFAULE_VECTOR_SIZE) : Vector<T>(size)
		{
			m_nTop = -1;
		}
		/// constructor
		Array(Array &array) { Copy(array); }
		/// constructor
		Array(T* pbuf, int n) { Copy(pbuf, n); }
		/// destructor
		~Array() { m_nTop = -1; }
		/// get the value at position i
		T& operator [](int i)
		{
			if (i < 0) {
				cout << "[warning] the index less than 0: i=" << i << endl;
				/*#ifdef _DEBUG*/
				//getch();
				/*#endif*/
				i = 0;
			}

			if (i > m_nTop)
				m_nTop = i;

			return this->Get(i);
		}
		/// Set Array number, to melloc enough memory.
		void SetNum(int n) { this->Get(n - 1); m_nTop = n - 1; }
		/// Get Array number
		int GetNum() const { return m_nTop + 1; }
		/// Add a value to the tail of array
		void Add(T t) { (*this)[m_nTop + 1] = t; }
		/// Add a value to the tail of array, example 'a.Add() = t'
		T& Add() { return (*this)[m_nTop + 1]; }
		/// Add a array to the tail of current array
		void Add(Array<T> &a) { for (int i = 0; i < a.GetNum(); i++) { Add(a[i]); } }
		/// Find a value and return the position
		int Find(T t) {
			for (int i = 0; i < GetNum(); i++) {
				if ((*this)[i] == t)
					return i;
			}
			return -1;
		}
		/// Get the value at the tail of array
		T& End() { return (*this)[m_nTop]; }
		/// Clean the array. Just set the top of array to -1 and donot release the memory
		void Clean() { m_nTop = -1; }
		/// Copy the array to current array
		void Copy(const Array<T> &array)
		{
			Vector<T>::Copy(array);
			m_nTop = array.m_nTop;
		}
		/// Copy the array to current array
		void Copy(const T* pbuf, int n)
		{
			Clean();
			for (int i = 0; i < n; i++)
				(*this)[i] = pbuf[i];
		}
		///  using memcpy to copy a array.
		void MemCopy(Array<T> &array)
		{
			Vector<T>::MemCopy(array);
			m_nTop = array.m_nTop;
		}
		/// overload operator =
		void operator = (const Array<T> &array) { Copy(array); }
		/// overload transite operator
		operator T* () { return this->GetBuffer(); }
		/// output the array
		void Output(ostream &os)
		{
			for (int i = 0; i < GetNum(); i++) {
				os << this->Get(i) << " ";
			}
		}
		/// input the array
		void Input(istream &is)
		{
			T t;
			while (is >> t) {
				Add(t);
			}
		}
		/// insert a value. Avoid repeating
		void Insert(T t)
		{
			for (int i = 0; i < GetNum(); i++) {
				if ((*this)[i] == t) {
					return;
				}
			}
			Add(t);
		}
		/// summate all the values in the array
		T Sum()
		{
			T sum = 0;
			for (int i = 0; i < GetNum(); i++) {
				sum += (*this)[i];
			}
			return sum;
		}
		/// Get the maximum value
		T Max(int &idx) {
			if (GetNum() == 0) {
				idx = -1;
				return 0;
			}

			idx = 0;
			for (int i = 1; i < GetNum(); i++) {
				if (this->Get(i) > this->Get(idx)) {
					idx = i;
				}
			}
			return this->Get(idx);
		}
		/// Get the minine value
		T Min(int &idx)
		{
			if (GetNum() == 0) {
				idx = -1;
				return 0;
			}

			idx = 0;
			for (int i = 1; i < GetNum(); i++) {
				if (this->Get(i) < this->Get(idx)) {
					idx = i;
				}
			}
			return this->Get(idx);
		}
	};

	/**
	* \class DArray
	* \date 2016-04-28
	* \brief 2-dimension array.
	* \author WangBin
	*/
	template <typename T>
	class DArray : public Vector<T>
	{
	public:
		int m_nXDim; ///< x-dim
		int m_nYDim; ///< y-dim
		/// constructor
		DArray(int x_size = DEFAULE_VECTOR_SIZE, int y_size = DEFAULE_VECTOR_SIZE) : Vector<T>(x_size*y_size)
		{
			m_nXDim = x_size;
			m_nYDim = y_size;
		}
		/// Get the value at position i row and j column
		T& Get(int i, int j)
		{
			return Vector<T>::Get(i*m_nYDim + j);
		}
	};

	/**
	* \class Stack
	* \date 2016-04-28
	* \brief A dynamic stack.
	* \author Wangbin
	*/
	template <typename T>
	class Stack : public Vector<T>
	{
	public:
		int m_nTop; ///< top of the stack
		/// constructor
		Stack(int size = DEFAULE_VECTOR_SIZE) : Vector<T>(size)
		{
			m_nTop = 0;
		}
		/// clean the stack. Donot release the memory
		void Clean() { m_nTop = 0; }
		/// Push a value into the stack
		void Push(T p_t)
		{
			this->Get(m_nTop) = p_t;
			m_nTop++;
		}
		/// Pop a value outof the stack
		bool Pop(T *p_pT)
		{
			if (m_nTop == 0)
				return false;

			if (p_pT != NULL)
				*p_pT = this->Get(m_nTop - 1);

			m_nTop--;
			return true;
		}
		/// Get the Top value of stack
		T Top()
		{
			if (m_nTop == 0)
				return this->Get(0);
			return this->Get(m_nTop - 1);
		}
		/// Get the number of the stack
		int GetNum() const { return m_nTop; }
	};

	/**
	* \class Queue
	* \brief  A queue based the dynamic memory mangement.
	* \date 2016-04-28
	* \author WangBin
	*
	*	This is not a circular quque.
	*	As ar result, the memory size will increase following the values added to the queue.
	*/
	template <typename T>
	class Queue : public Vector<T>
	{
	public:
		int m_nTop; ///< the top of the queue
		int m_nBottom; ///< the bottom of the queue
		/// constructor
		Queue(int size = DEFAULE_VECTOR_SIZE) : Vector<T>(size)
		{
			m_nTop = 0;
			m_nBottom = 0;
		}
		/// clean the queue. Donot release the memory
		void Clean() { m_nTop = 0; m_nBottom = 0; }
		/// Add a value into queue
		void In(T p_t)
		{
			this->Get(m_nTop) = p_t;
			m_nTop++;
		}
		/// Move a value outof queue
		bool Out(T *p_pT)
		{
			if (m_nBottom == m_nTop)
				return false;
			if (p_pT)
				*p_pT = this->Get(m_nBottom);
			m_nBottom++;
			return true;
		}
		/// Return if the queue is empty
		bool IsEmpty()
		{
			return m_nBottom == m_nTop;
		}
		/// Return the value number
		int GetNum() { return m_nTop - m_nBottom; }
	};


	/**
	* \class CirQueue
	* \date 2016-04-28
	* \brief Circular queue
	* \author WangBin
	*
	*  For circular queue, we need indicate the size of a queue beforhead.
	*/
	template <typename T>
	class CirQueue : Vector<T>
	{
	private:
		int nSize; ///< the size of queue
		bool bFull; ///< if the queue is full
	public:
		int nHead; ///< the head of queue
		int nTail; ///< the tail of queue
	public:
		/// constructor
		CirQueue(int size = 10) :Vector<T>(size)
		{
			nHead = 0;
			nTail = 0;
			bFull = false;
			nSize = size;
		}
		/// Init the queue
		void Init(int size)
		{
			Clean();
			this->Get(size);
			nSize = size;
		}
		/// Clean the queue, don't release the memory
		void Clean()
		{
			nHead = 0;
			nTail = 0;
			bFull = false;
		}
		/// Add a value into queue
		void In(T t)
		{
			if (!bFull) {
				this->Get(nTail) = t;
				nTail = (nTail + 1) % nSize;
				if (nTail == nHead)
					bFull = true;
			}
			else {
				Out();
				In(t);
			}
		}
		/// Remove a value outof queue
		bool Out(T *t = NULL)
		{
			if (!bFull) {
				if (nHead == nTail) {
					// empty
					return false;
				}
				else {
					if (t) *t = this->Get(nHead);
					nHead = (nHead + 1) % nSize;
					return true;
				}
			}
			else {
				if (t) *t = this->Get(nHead);
				nHead = (nHead + 1) % nSize;
				bFull = false;
				return true;
			}
		}
		/// if the queue is empty
		bool IsEmpty() { return (!bFull) && nHead == nTail; }
		/// if the queue is full
		bool IsFull() { return bFull; }
		/// Get number of values
		int GetNum() { return (bFull) ? nSize : max(nTail - nHead, nTail + nSize - nHead); }
		/// Get the buffer size of queue
		int GetSize() { return nSize; }
		/// Summary all the values if queue
		T GetSum()
		{
			T sum = 0;
			if (bFull) {
				for (int i = 0; i < nSize; i++)
					sum += this->Get(i);
			}
			else {
				for (int i = nHead; i != nTail; i = (i + 1) % nSize)
					sum += this->Get(i);
			}
			return sum;
		}
	};

	/// [Vec-function] sqrt(v*v^T);
	template <typename T>
	double VecNorm(T *pVec, int len)
	{
		double sum = 0;
		for (int i = 0; i < len; i++) {
			sum += pVec[i] * pVec[i];
		}
		return sqrt(sum);
	}
	/// [Vec-function] v1*v2^T
	template <typename T>
	double VecDot(T *pVec1, T *pVec2, int len)
	{
		double sum = 0;
		for (int i = 0; i < len; i++) {
			sum += pVec1[i] * pVec2[i];
		}
		return sum;
	}
	/// [Vec-function] |v1-v2|
	template <typename T>
	double VecDiff(T *pVec1, T*pVec2, int len)
	{
		double sum = 0;
		for (int i = 0; i < len; i++) {
			sum += pow(pVec1[i] - pVec2[i], 2);
		}
		return sqrt(sum);
	}
	/// [Vec-function] the cos of the angle of v1 and v2
	template <typename T>
	double VecAngle(T *pVec1, T* pVec2, int len)
	{
		double sum = Dot(pVec1, pVec2, len);
		return sum / Norm(pVec1, len) / Norm(pVec2, len);
	}
	/// [Vec-function] transform the matlab-style vector to a array, such as [1,3:1:7,9] => 1,3,4,5,6,7,9
	template <typename T>
	void VecUnfold(const char *pStr, Array<T> &a)
	{
		a.Clean();

		if (pStr == NULL)
			return;

		Array<char*> aStr;
		char *pStrDup = strdup(pStr);
		char *p = strtok(pStrDup, "[], ");
		while (p) {
			aStr.Add() = strdup(p);
			p = strtok(NULL, "[], ");
		}
		free(pStrDup);

		for (int i = 0; i < aStr.GetNum(); i++) {
			Array<double> d;
			char *p = strtok(aStr[i], ": ");
			while (p) {
				d.Add((T)atof(p));
				p = strtok(NULL, ": ");
			}

			double t = 0;
			switch (d.GetNum()) {
			case 1: a.Add(d[0]); break;
			case 2:
				for (t = d[0]; t <= d[1]; t += 1)
					a.Add(t);
				break;
			case 3:
				for (t = d[0]; t <= d[2]; t += d[1])
					a.Add(t);
				break;
			default:
				cout << "Error vector expression!! =>" << pStr << endl;
				exit(1);
			}

		}

		for (int i = 0; i < aStr.GetNum(); i++) {
			free(aStr[i]);
		}
	}

	/// comparing function for increase sort.
	template <typename T>
	int  Compar_Inc(const T& a, const T& b)
	{
		if (a > b) return 1;
		else return -1;
	}
	/// comparing function for decrease sort
	template <typename T>
	int Compar_Dec(const T& a, const T& b)
	{
		if (a > b) return -1;
		else return 1;
	}

	/// Quick sork
	/*
		\param [in] p the input array
		\param [in] low the smallest index needed to be sorted, (commonly = 0)
		\param [in] high the largest index needed to be sorted, (commonly = the length - 1)
		\param [in] compar the compar function. Set to 'Compar_Inc' for increase and 'Compar_Dec' for decrease
	*/
	template <typename T>
	void Qsort(T *p, int low, int high, int(*compar)(const T&, const T&) = Compar_Inc)
	{
		if (low >= high)
		{
			return;
		}
		int first = low;
		int last = high;
		T key = p[first];

		while (first < last)
		{
			while (first < last && compar(p[last], key) >= 0)
			{
				--last;
			}

			p[first] = p[last];

			while (first < last && compar(p[first], key) <= 0)
			{
				++first;
			}

			p[last] = p[first];

		}
		p[first] = key;
		Qsort(p, low, first - 1, compar);
		Qsort(p, first + 1, high, compar);
	}
	/// Quick sork, redefine for class Array
	template <typename T>
	void Qsort(Array<T> &a, int(*compar)(const T&, const T&) = Compar_Inc)
	{
		Qsort(a.GetBuffer(), 0, a.GetNum() - 1, compar);
	}


	/** @} */
}

#endif




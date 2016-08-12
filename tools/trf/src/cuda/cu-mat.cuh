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
* \brief	For cuda class, the operations to allocate or free memory are all __host__ only,\n
			and the operations depending on the memory read or write are __device__ only. \n
			The others are __host__ and __device__.
*/


#ifndef _CU_MAT_CUH_
#define _CU_MAT_CUH_
#include "cu-def.cuh"
#include "wb-mat.h"
#include "wb-vector.h"

namespace cu
{

	template <class T> class VecShell;
	template <class T> class MatShell;
	template <class T> class Vec;
	template <class T> class Mat;


	template <class T>
	/**
	* \class
	* \brief Vector Shell. The class donot new any memory and the buffer is assigned out-of the class
	*/
	class VecShell
	{
		/// define the friend function
		template <class T> friend bool VecEqual(VecShell<T> &v1, VecShell<T> &v2);
		template <class T> friend void VecAdd(VecShell<T> &res, VecShell<T> &v1, VecShell<T> &v2);
		template <class T> friend T VecDot(VecShell<T> &v1, VecShell<T> &v2);
		template <class T> friend T MatVec2(MatShell<T> &m, VecShell<T> &v1, VecShell<T> &v2);

	protected:
		T *m_pBuf; ///< buf pointer
		int m_nSize; ///< buf size
	public:
		__host__ __device__  VecShell() : m_pBuf(NULL), m_nSize(0) {};
		__host__ __device__  VecShell(T *p, int size) : m_pBuf(p), m_nSize(size) {};
		__host__ __device__ T* GetBuf() const { return m_pBuf; }
		__host__ __device__ int GetSize() const { return m_nSize; }
		__host__ __device__ int ByteSize() const { return GetSize() * sizeof(T); }
		__host__ __device__ VecShell<T> GetSub(int nPos, int nLen) { return VecShell<T>(m_pBuf + nPos, nLen); }
		__host__ __device__ void Reset(T *p, int size) { m_pBuf = p; m_nSize = size; }
		__device__ void Fill(T v);
		__device__ T& operator [] (int i);
		__device__ void operator = (VecShell &v);
		__device__ void operator += (VecShell &v);
		__device__ void operator -= (VecShell &v);
		__device__ void operator *= (T n);
		__device__ void operator /= (T n);
		__device__ bool operator == (VecShell &v);
		//operator T*() const { return m_pBuf; }
		__host__ void CopyTo(wb::Vec<T> &v);
		__host__ void CopyTo(wb::VecShell<T> v);
	};

	template <class T>
	/**
	* \class
	* \brief Vector. Melloc the buffer in this class
	*/
	class Vec : public VecShell<T>
	{
	private:
		int m_nBufSize; ///< Save the real size of the buffer. To avoid del/new memory frequently
	public:
		__host__ Vec() :m_nBufSize(0) { Reset(); }
		__host__ Vec(int size) :m_nBufSize(0) { Reset(size); }
		__host__ Vec(wb::VecShell<T> v) : m_nBufSize(0) { Copy(v); }
		__host__ ~Vec() { /*Reset();*/ }
		__host__ void Reset(int size = 0);
		__host__ void Copy(wb::VecShell<T> v);
		__host__ void Copy(const wb::Array<T> &a);
		
	};


	template <class T>
	/**
	* \class
	* \brief Matrix Shell, Don't alloc/release any memory in this class
	*/
	class MatShell
	{
	protected:
		T *m_pBuf; ///< buf pointer
		int m_nRow;
		int m_nCol;
	public:
		__host__ __device__ MatShell() : m_pBuf(NULL), m_nRow(0), m_nCol(0) {};
		__host__ __device__ MatShell(T *pbuf, int row, int col) :m_pBuf(pbuf), m_nRow(row), m_nCol(col) {};
		__device__ T* GetBuf() const { return m_pBuf; }
		__device__ T& Get(unsigned int i, unsigned int j) { return m_pBuf[i*m_nCol + j]; }
		__host__ __device__ int GetSize() const { return m_nRow*m_nCol; }
		__host__ __device__ int ByteSize() const { return GetSize() * sizeof(T); }
		__host__ __device__ int GetRow() const { return m_nRow; }
		__host__ __device__ int GetCol() const { return m_nCol; }
		__host__ __device__ void Reset(T *pbuf, int row, int col) { m_pBuf = pbuf; m_nRow = row; m_nCol = col; }
		__host__ __device__ VecShell<T> operator [] (int i) { return VecShell<T>(m_pBuf + i*m_nCol, m_nCol); }
		__host__ __device__ operator T* () { return m_pBuf; }
		__device__ bool operator== (MatShell &m);
		__host__ void CopyTo(wb::Mat<T> &m);
	};

	template <class T>
	/**
	* \class
	* \brief Matrix. Alloc/Release memory
	*/
	class Mat : public MatShell<T>
	{
	private:
		int m_nBufSize; ///< Save the real size of the buffer. To avoid del/new memory frequently
	public:
		__host__ Mat() : m_nBufSize(0) { Reset(); }
		__host__ Mat(int row, int col) : m_nBufSize(0) { Reset(row, col); }
		__host__ ~Mat() { /*Reset();*/ }
		__host__ void Reset(int row = 0, int col = 0);
		__host__ void Copy(wb::MatShell<T> &m);
		
	};

	template <class T>
	/**
	* \class
	* \brief Shell of 3D matrix.
	*/
	class Mat3dShell
	{
	protected:
		T* m_pBuf;
		int m_nXDim;
		int m_nYDim;
		int m_nZDim;
	public:
		__host__ __device__ Mat3dShell() :m_pBuf(NULL), m_nXDim(0), m_nYDim(0), m_nZDim(0) {};
		__host__ __device__ Mat3dShell(T* p, int xdim, int ydim, int zdim) :m_pBuf(p), m_nXDim(xdim), m_nYDim(ydim), m_nZDim(zdim) {};
		__host__ __device__ T* GetBuf() const { return m_pBuf; }
		__device__ T& Get(int x, int y, int z) { return m_pBuf[x*m_nYDim*m_nZDim + y*m_nZDim + z]; }
		__host__ __device__ int GetSize() const { return m_nXDim * m_nYDim * m_nZDim; }
		__host__ __device__ int ByteSize() const { return GetSize() * sizeof(T); }
		__host__ __device__ int GetXDim() const { return m_nXDim; }
		__host__ __device__ int GetYDim() const { return m_nYDim; }
		__host__ __device__ int GetZDim() const { return m_nZDim; }
		__host__ __device__ void Reset(T* p, int xdim, int ydim, int zdim) { m_pBuf = p; m_nXDim = xdim; m_nYDim = ydim; m_nZDim = zdim; }
		__device__ MatShell<T> operator[] (int x) { return MatShell<T>(m_pBuf + x*m_nYDim*m_nZDim, m_nYDim, m_nZDim); }
	};

	template <class T>
	/**
	* \class
	* \brief 3d matrix
	*/
	class Mat3d : public Mat3dShell<T>
	{
	private:
		int m_nBufSize; ///< Save the real size of the buffer. To avoid del/new memory frequently
	public:
		__host__ Mat3d() : m_nBufSize(0) { Reset(); }
		__host__ Mat3d(int xdim, int ydim, int zdim) : m_nBufSize(0) { Reset(xdim, ydim, zdim); }
		__host__ ~Mat3d() {/* Reset(); */ }
		__host__ void Reset(int xdim = 0, int ydim = 0, int zdim = 0);
		__host__ void Copy(wb::Mat3dShell<T> &m);
	};

	/************************************************************************/
	/* VecShell                                                             */
	/************************************************************************/

	template <class T>
	__device__ void VecShell<T>::Fill(T v)
	{
		if (!m_pBuf) {
			return;
		}
		for (int i = 0; i < m_nSize; i++) {
			m_pBuf[i] = v;
		}
	}
	template <class T>
	__device__ T& VecShell<T>::operator[](int i)
	{
		return m_pBuf[i];
	}
	template <class T>
	__device__ void VecShell<T>::operator=(VecShell<T> &v)
	{
		if (v.GetSize() != GetSize()) {
			lout_error("[VecShell] op=: the size is not equal (" << v.GetSize() << ")(" << GetSize() << ")");
		}
		memcpy(m_pBuf, v.m_pBuf, sizeof(T)*m_nSize);
	}
	template <class T>
	__device__ void VecShell<T>::operator+=(VecShell<T> &v)
	{
		if (v.GetSize() != GetSize()) {
			lout_error("[VecShell] op+=: the size is not equal (" << v.GetSize() << ")(" << GetSize() << ")");
		}
		for (int i = 0; i < GetSize(); i++) {
			m_pBuf[i] += v.m_pBuf[i];
		}
	}
	template <class T>
	__device__ void VecShell<T>::operator-=(VecShell<T> &v)
	{
		if (v.GetSize() != GetSize()) {
			lout_error("[VecShell] op+=: the size is not equal (" << v.GetSize() << ")(" << GetSize() << ")");
		}
		for (int i = 0; i < GetSize(); i++) {
			m_pBuf[i] -= v.m_pBuf[i];
		}
	}
	template <class T>
	__device__ void VecShell<T>::operator*=(T n)
	{
		for (int i = 0; i < GetSize(); i++) {
			m_pBuf[i] *= n;
		}
	}
	template <class T>
	__device__ void VecShell<T>::operator/=(T n)
	{
		for (int i = 0; i < GetSize(); i++) {
			m_pBuf[i] /= n;
		}
	}
	template <class T>
	__device__ bool VecShell<T>::operator==(VecShell<T> &v)
	{
		if (m_nSize != v.m_nSize)
			return false;
		for (int i = 0; i < m_nSize; i++) {
			if (m_pBuf[i] != v[i]) {
				return false;
			}
		}
		return true;
	}

	/************************************************************************/
	/* Vec                                                                  */
	/************************************************************************/
	template <class T>
	__host__ void Vec<T>::Reset(int size /* = 0 */)
	{
		if (size == 0) {
			// Clean buffer
			CUDA_SAFE_FREE(m_pBuf);
			m_nSize = 0;
			m_nBufSize = 0;
			return;
		}

		if (size <= m_nBufSize) { // donot re-alloc memory
			m_nSize = size;
		}
		else { // Re-alloc
			T *p;
			CUDA_CALL(cudaMalloc(&p, sizeof(T)*size));
			if (m_pBuf) {
				CUDA_CALL(cudaMemcpy(p, m_pBuf, sizeof(T)*m_nSize, cudaMemcpyDeviceToDevice));
				CUDA_SAFE_FREE(m_pBuf);
			}
			m_pBuf = p;
			m_nSize = size;
			m_nBufSize = size;
		}
	}
	template <class T>
	__host__ void Vec<T>::Copy(wb::VecShell<T> v)
	{
		Reset(v.GetSize());
		CUDA_CALL(cudaMemcpy(m_pBuf, v.GetBuf(), v.ByteSize(), cudaMemcpyHostToDevice));
	}
	template <class T>
	__host__ void Vec<T>::Copy(const wb::Array<T> &a)
	{
		Reset(a.GetNum());
		CUDA_CALL(cudaMemcpy(m_pBuf, a.GetBuffer(), a.GetNum() * sizeof(T), cudaMemcpyHostToDevice))
	}
	template <class T>
	__host__ void VecShell<T>::CopyTo(wb::Vec<T> &v)
	{
		v.Reset(GetSize());
		CUDA_CALL(cudaMemcpy(v.GetBuf(), m_pBuf, ByteSize(), cudaMemcpyDeviceToHost));
	}
	template <class T>
	__host__ void VecShell<T>::CopyTo(wb::VecShell<T> v)
	{
		CUDA_CALL(cudaMemcpy(v.GetBuf(), m_pBuf, min(v.ByteSize(), ByteSize()), cudaMemcpyDeviceToHost));
	}

	/************************************************************************/
	/* MatShell                                                             */
	/************************************************************************/

	template <class T>
	__device__ bool MatShell<T>::operator==(MatShell<T> &m)
	{
		if (m_nRow != m.m_nRow || m_nCol != m.m_nCol)
			return false;
		for (int i = 0; i < m_nRow*m_nCol; i++) {
			if (m_pBuf[i] != m.m_pBuf[i])
				return false;
		}
		return true;
	}

	/************************************************************************/
	/* Mat                                                                  */
	/************************************************************************/

	template <class T>
	void Mat<T>::Reset(int row /* = 0 */, int col /* = 0 */)
	{
		if (row * col == 0) {
			// Clean buffer
			CUDA_SAFE_FREE(m_pBuf);
			m_nRow = 0;
			m_nCol = 0;
			m_nBufSize = 0;
			return;
		}

		int size = row * col;
		if (size <= m_nBufSize) {
			m_nRow = row;
			m_nCol = col;
		}
		else {
			T *p;
			CUDA_CALL(cudaMalloc(&p, sizeof(T)*size));
			if (m_pBuf) {
				CUDA_CALL(cudaMemcpy(p, m_pBuf, sizeof(T)*m_nRow*m_nCol, cudaMemcpyDeviceToDevice));
				//memcpy(p, m_pBuf, sizeof(T)*m_nRow*m_nCol);
				CUDA_SAFE_FREE(m_pBuf);
			}
			m_pBuf = p;
			m_nRow = row;
			m_nCol = col;
			m_nBufSize = size;
		}
	}
	template <class T>
	void Mat<T>::Copy(wb::MatShell<T> &m)
	{
		Reset(m.GetRow(), m.GetCol());
		CUDA_CALL(cudaMemcpy(m_pBuf, m.GetBuf(), m.ByteSize(), cudaMemcpyHostToDevice));
		//memcpy(m_pBuf, m.GetBuf(), sizeof(T)*m_nRow*m_nCol);
	}
	template <class T>
	__host__ void MatShell<T>::CopyTo(wb::Mat<T> &m)
	{
		m.Reset(m_nRow, m_nCol);
		CUDA_CALL(cudaMemcpy(m.GetBuf(), m_pBuf, ByteSize(), cudaMemcpyDeviceToHost));
	}


	template <class T>
	void Mat3d<T>::Reset(int xdim/* =0 */, int ydim/* =0 */, int zdim/* =0 */)
	{
		if (xdim*ydim*zdim == 0) {
			// Clean buffer
			CUDA_SAFE_FREE(m_pBuf);
			m_nXDim = 0;
			m_nYDim = 0;
			m_nZDim = 0;
			m_nBufSize = 0;
			return;
		}

		int size = xdim * ydim * zdim;
		if (size <= m_nBufSize) {
			m_nXDim = xdim;
			m_nYDim = ydim;
			m_nZDim = zdim;
		}
		else {
			T *p;
			CUDA_CALL(cudaMalloc(&p, sizeof(T)*size));
			if (m_pBuf) {
				CUDA_CALL(cudaMemcpy(p, m_pBuf, sizeof(T)*m_nXDim*m_nYDim*m_nZDim, cudaMemcpyDeviceToDevice));
				CUDA_SAFE_FREE(m_pBuf);
			}
			m_pBuf = p;
			m_nXDim = xdim;
			m_nYDim = ydim;
			m_nZDim = zdim;
			m_nBufSize = size;
		}
	}
	template <class T>
	void Mat3d<T>::Copy(wb::Mat3dShell<T> &m)
	{
		Reset(m.GetXDim(), m.GetYDim(), m.GetZDim());
		CUDA_CALL(cudaMemcpy(m_pBuf, m.GetBuf(), m.ByteSize(), cudaMemcpyHostToDevice));
		//memcpy(m_pBuf, m.GetBuf(), sizeof(T)*m.GetSize());
	}
}

#endif
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


#ifndef _WB_MAT_H_
#define _WB_MAT_H_
#include "wb-vector.h"
#include "wb-log.h"
#include "wb-file.h"

namespace wb
{

	template <class T> class VecShell;
	template <class T> class MatShell;
	template <class T> class Vec;
	template <class T> class Mat;

	template <class T> bool VecEqual(VecShell<T> &v1, VecShell<T> &v2);
	template <class T> void VecAdd(VecShell<T> &res, VecShell<T> &v1, VecShell<T> &v2);
	template <class T> T VecDot(VecShell<T> &v1, VecShell<T> &v2);
	template <class T> T MatVec2(MatShell<T> &m, VecShell<T> &v1, VecShell<T> &v2);
	template <class T> T MatVec2(MatShell<T> &m, VecShell<T> &v1, VecShell<T> &v2);


	template <class T>
	/**
	 * \class
	 * \brief Vector Shell. The class donot new any memory and the buffer is assigned out-of the class
	 */
	class VecShell
	{
		/// define the friend function
		friend bool VecEqual<>(VecShell<T> &v1, VecShell<T> &v2);
		friend void VecAdd<>(VecShell<T> &res, VecShell<T> &v1, VecShell<T> &v2);
		friend T VecDot<>(VecShell<T> &v1, VecShell<T> &v2);
		friend T MatVec2<>(MatShell<T> &m, VecShell<T> &v1, VecShell<T> &v2);

	protected:
		T *m_pBuf; ///< buf pointer
		int m_nSize; ///< buf size
	public:
		VecShell() : m_pBuf(NULL), m_nSize(0) {};
		VecShell(T *p, int size) : m_pBuf(p), m_nSize(size) {};
		void Fill(T v);
		T* GetBuf() const { return m_pBuf; }
		int GetSize() const { return m_nSize; }
		int ByteSize() const { return GetSize() * sizeof(T); }
		VecShell<T> GetSub(int nPos, int nLen) { return VecShell<T>(m_pBuf + nPos, nLen); }
		void Reset(T *p, int size) { m_pBuf = p; m_nSize = size; }
		T& operator [] (int i);
		void operator = (VecShell v);
		void operator += (VecShell v);
		void operator -= (VecShell v);
		void operator *= (T n);
		void operator /= (T n);
		bool operator == (VecShell v);
		//operator T*() const { return m_pBuf; }
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
		Vec():m_nBufSize(0) { Reset(); }
		Vec(int size):m_nBufSize(0) { Reset(size); }
		~Vec() { Reset(); }
		void Reset(int size = 0);
		void Copy(VecShell<T> v);
	};


	template <class T>
	/**
	 * \class
	 * \brief Matrix Shell, Don't alloc/release any memory in this class
	 */
	class MatShell
	{
		friend T MatVec2<>(MatShell<T> &m, VecShell<T> &v1, VecShell<T> &v2);
	protected:
		T *m_pBuf; ///< buf pointer
		int m_nRow;
		int m_nCol;
	public:
		MatShell() : m_pBuf(NULL), m_nRow(0), m_nCol(0) {};
		MatShell(T *pbuf, int row, int col) :m_pBuf(pbuf), m_nRow(row), m_nCol(col) {};
		void Fill(T v);
		T* GetBuf() const { return m_pBuf; }
		T& Get(unsigned int i, unsigned int j) { return m_pBuf[i*m_nCol + j]; }
		int GetSize() const { return m_nRow*m_nCol; }
		int ByteSize() const { return GetSize() * sizeof(T); }
		int GetRow() const { return m_nRow; }
		int GetCol() const { return m_nCol; }
		void Reset(T *pbuf, int row, int col) { m_pBuf = pbuf; m_nRow = row; m_nCol = col; }
		VecShell<T> operator [] (int i) { return VecShell<T>(m_pBuf + i*m_nCol, m_nCol);}
		operator T* () { return m_pBuf; }
		bool operator== (MatShell &m);
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
		Mat() : m_nBufSize(0) { Reset(); }
		Mat(int row, int col) : m_nBufSize(0) { Reset(row, col); }
		~Mat() { Reset(); }
		void Reset(int row = 0, int col = 0);
		void Copy(MatShell<T> &m);
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
		Mat3dShell() :m_pBuf(NULL), m_nXDim(0), m_nYDim(0), m_nZDim(0) {};
		Mat3dShell(T* p, int xdim, int ydim, int zdim) :m_pBuf(p), m_nXDim(xdim), m_nYDim(ydim), m_nZDim(zdim) {};
		void Fill(T v);
		T* GetBuf() const { return m_pBuf; }
		T& Get(int x, int y, int z) { return m_pBuf[x*m_nYDim*m_nZDim + y*m_nZDim + z]; }
		int GetSize() const { return m_nXDim * m_nYDim * m_nZDim; }
		int ByteSize() const { return GetSize() * sizeof(T); }
		int GetXDim() const { return m_nXDim; }
		int GetYDim() const { return m_nYDim; }
		int GetZDim() const { return m_nZDim; }
		void Reset(T* p, int xdim, int ydim, int zdim) { m_pBuf = p; m_nXDim = xdim; m_nYDim = ydim; m_nZDim = zdim; }
		MatShell<T> operator[] (int x) { return MatShell<T>(m_pBuf + x*m_nYDim*m_nZDim, m_nYDim, m_nZDim); }
		void Write(File &file);
		void Read(File &file);
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
		Mat3d() : m_nBufSize(0) { Reset(); }
		Mat3d(int xdim, int ydim, int zdim) : m_nBufSize(0) { Reset(xdim, ydim, zdim); }
		~Mat3d() { Reset(); }
		void Reset(int xdim=0, int ydim=0, int zdim=0);
		void Copy(Mat3dShell<T> &m);
	};



	/************************************************************************/
	/*  mat * vec / vec * vec                                               */
	/************************************************************************/

	template <class T>
	/// calculate V==V
	bool VecEqual(VecShell<T> &v1, VecShell<T> &v2)
	{
		if (v1.GetSize() != v2.GetSize())
			return false;
		for (int i = 0; i < v1.GetSize(); i++) {
			if (v1[i] != v2[i]) {
				return false;
			}
		}
		return true;
	}
	template <class T>
	/// calculate V + V
	void VecAdd(VecShell<T> &res, VecShell<T> &v1, VecShell<T> &v2)
	{
		if (v1.GetSize() != v2.GetSize() || res.GetSize() != v1.GetSize()) {
			lout_error("[VecAdd] Vec Size are not equal: v1.size=" 
				<< v1.GetSize() << " v2.size=" << v2.GetSize() << " res.size=" << res.GetSize());
		}
		for (int i = 0; i < res.GetSize(); i++) {
			res.m_pBuf[i] = v1.m_pBuf[i] + v2.m_pBuf[i];
		}
	}
	template <class T>
	/// calculate V*V
	T VecDot(VecShell<T> &v1, VecShell<T> &v2)
	{
#ifdef _DEBUG
		if (v1.GetSize() != v2.GetSize()) {
			lout_error("[VecDot] v1.size(" << v1.GetSize() << ") != v2.size(" << v2.GetSize() << ")");
		}
#endif
		T sum = 0;
		for (int i = 0; i < v1.GetSize(); i++) {
			sum += v1.m_pBuf[i] * v2.m_pBuf[i];
		}
		return sum;
	}

	template <class T>
	/// calculate V1*M*V2
	T MatVec2(MatShell<T> &m, VecShell<T> &v1, VecShell<T> &v2)
	{
#ifdef _DEBUG
		if (v1.GetSize() != m.GetRow()) {
			lout_error("[MatVec2] v1.size(" << v1.GetSize() << ") != m.row(" << m.GetRow() << ")");
		}
		if (v2.GetSize() != m.GetCol()) {
			lout_error("[MatVec2] m.col(" << m.GetCol() << ") != v2.size(" << v2.GetSize() << ")");
		}
#endif
		T sum = 0;
		for (int i = 0; i < m.GetRow(); i++) {
			if (v1.m_pBuf[i] == 0)
				continue;
			for (int j = 0; j < m.GetCol(); j++) {
				if (v2.m_pBuf[j] == 0)
					continue;
				sum += v1.m_pBuf[i] * m.Get(i,j) * v2.m_pBuf[j];
			}
		}
		return sum;
	}

	/************************************************************************/
	/* VecShell                                                             */
	/************************************************************************/

	template <class T>
	void VecShell<T>::Fill(T v)
	{
		if (!m_pBuf) {
			return;
		}
		for (int i = 0; i < m_nSize; i++) {
			m_pBuf[i] = v;
		}
	}
	template <class T>
	T& VecShell<T>::operator[](int i)
	{
#ifdef _DEBUG
		if (!m_pBuf) {
			lout_error("[Vec] op[]: buffer = NULL");
		}
		if (i < 0 || i >= m_nSize) {
			lout_error("[Vec] op[] index i(" << i << ") over the size(" << m_nSize << ")");
		}
#endif
		return m_pBuf[i];
	}
	template <class T>
	void VecShell<T>::operator=(VecShell<T> v)
	{
		if (v.GetSize() != GetSize()) {
			lout_error("[VecShell] op=: the size is not equal (" << v.GetSize() << ")(" << GetSize() << ")");
		}
		memcpy(m_pBuf, v.m_pBuf, sizeof(T)*m_nSize);
	}
	template <class T>
	void VecShell<T>::operator+=(VecShell<T> v)
	{
		if (v.GetSize() != GetSize()) {
			lout_error("[VecShell] op+=: the size is not equal (" << v.GetSize() << ")(" << GetSize() << ")");
		}
		for (int i = 0; i < GetSize(); i++) {
			m_pBuf[i] += v.m_pBuf[i];
		}
	}
	template <class T>
	void VecShell<T>::operator-=(VecShell<T> v)
	{
		if (v.GetSize() != GetSize()) {
			lout_error("[VecShell] op+=: the size is not equal (" << v.GetSize() << ")(" << GetSize() << ")");
		}
		for (int i = 0; i < GetSize(); i++) {
			m_pBuf[i] -= v.m_pBuf[i];
		}
	}
	template <class T>
	void VecShell<T>::operator*=(T n)
	{
		for (int i = 0; i < GetSize(); i++) {
			m_pBuf[i] *= n;
		}
	}
	template <class T>
	void VecShell<T>::operator/=(T n)
	{
		for (int i = 0; i < GetSize(); i++) {
			m_pBuf[i] /= n;
		}
	}
	template <class T>
	bool VecShell<T>::operator==(VecShell<T> v)
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
	void Vec<T>::Reset(int size /* = 0 */)
	{
		if (size == 0) {
			// Clean buffer
			SAFE_DELETE_ARRAY(this->m_pBuf);
			this->m_nSize = 0;
			this->m_nBufSize = 0;
			return;
		}

		if (size <= this->m_nBufSize) { // donot re-alloc memory
			this->m_nSize = size;
		}
		else { // Re-alloc
			T *p = new T[size];
			if (p == NULL) {
				lout_error("[Vec] Reset: new buffer error!");
			}
			memcpy(p, this->m_pBuf, sizeof(T)*this->m_nSize);
			SAFE_DELETE_ARRAY(this->m_pBuf);
			this->m_pBuf = p;
			this->m_nSize = size;
			this->m_nBufSize = size;
		}
	}
	template <class T>
	void Vec<T>::Copy(VecShell<T> v)
	{
		Reset(v.GetSize());
		memcpy(this->m_pBuf, v.GetBuf(), sizeof(T)*v.GetSize());
	}

	/************************************************************************/
	/* MatShell                                                             */
	/************************************************************************/

	template <class T>
	void MatShell<T>::Fill(T v)
	{
		if (!m_pBuf) {
			lout_error("[Mat] Fill: buffer = NULL");
		}
		for (int i = 0; i < m_nRow*m_nCol; i++) {
			m_pBuf[i] = v;
		}
	}
	template <class T>
	bool MatShell<T>::operator==(MatShell<T> &m)
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
			SAFE_DELETE_ARRAY(this->m_pBuf);
			this->m_nRow = 0;
			this->m_nCol = 0;
			this->m_nBufSize = 0;
			return;
		}

		int size = row * col;
		if (size <= this->m_nBufSize) {
			this->m_nRow = row;
			this->m_nCol = col;
		}
		else {
			T *p = new T[size];
			if (p == NULL) {
				lout_error("[Mat] Reset: new buffer error!");
			}
			memcpy(p, this->m_pBuf, sizeof(T)*this->m_nRow*this->m_nCol);
			SAFE_DELETE_ARRAY(this->m_pBuf);
			this->m_pBuf = p;
			this->m_nRow = row;
			this->m_nCol = col;
			this->m_nBufSize = size;
		}
	}
	template <class T>
	void Mat<T>::Copy(MatShell<T> &m)
	{
		Reset(m.GetRow(), m.GetCol());
		memcpy(this->m_pBuf, m.GetBuf(), sizeof(T)*this->m_nRow*this->m_nCol);
	}

	template <class T>
	void Mat3dShell<T>::Fill(T v)
	{
		if (!m_pBuf) {
			lout_error("[Mat3d] Fill: buffer = NULL");
		}
		for (int i = 0; i < GetSize(); i++) {
			m_pBuf[i] = v;
		}
	}
	template <class T>
	void Mat3dShell<T>::Write(File &file)
	{
		ofstream os(file.fp);
		int x, y, z;
		for (x = 0; x < m_nXDim; x++) {
			for (y = 0; y < m_nYDim; y++) {
				os << "[";
				for (z = 0; z < m_nZDim-1; z++)
					os<<Get(x, y, z)<<" ";
				os << Get(x, y, z) << "]";
			}
			os << endl;
		}
	}
	template <class T>
	void Mat3dShell<T>::Read(File &file)
	{
		ifstream is(file.fp);
		int x, y, z;
		char c;
		for (x = 0; x < m_nXDim; x++) {
			for (y = 0; y < m_nYDim; y++) {
				is >> c;
				for (z = 0; z < m_nZDim - 1; z++)
					is >> Get(x, y, z);
				is >> Get(x, y, z) >> c;
			}
			// ����
			file.Scanf("\n");
		}
	}
	template <class T>
	void Mat3d<T>::Reset(int xdim/* =0 */, int ydim/* =0 */, int zdim/* =0 */)
	{
		if (xdim*ydim*zdim == 0) {
			// Clean buffer
			SAFE_DELETE_ARRAY(this->m_pBuf);
			this->m_nXDim = 0;
			this->m_nYDim = 0;
			this->m_nZDim = 0;
			this->m_nBufSize = 0;
			return;
		}

		int size = xdim * ydim * zdim;
		if (size <= this->m_nBufSize) {
			this->m_nXDim = xdim;
			this->m_nYDim = ydim;
			this->m_nZDim = zdim;
		}
		else {
			T *p = new T[size];
			if (p == NULL) {
				lout_error("[Mat] Reset: new buffer error!");
			}
			memcpy(p, this->m_pBuf, sizeof(T)*this->GetSize());
			SAFE_DELETE_ARRAY(this->m_pBuf);
			this->m_pBuf = p;
			this->m_nXDim = xdim;
			this->m_nYDim = ydim;
			this->m_nZDim = zdim;
			this->m_nBufSize = size;
		}
	}
	template <class T>
	void Mat3d<T>::Copy(Mat3dShell<T> &m)
	{
		Reset(m.GetXDim(), m.GetYDim(), m.GetZDim());
		memcpy(this->m_pBuf, m.GetBuf(), sizeof(T)*m.GetSize());
	}
}

#endif

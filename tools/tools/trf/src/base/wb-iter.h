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


#ifndef _WB_ITER_H_
#define _WB_ITER_H_
#include "wb-vector.h"

namespace wb
{
	namespace iter
	{
		template <typename T>
		class Obj
		{
		public:
			virtual void Reset() = 0;
			virtual bool Next(T &t) = 0;
		};

		template <typename T>
		class Line : public Obj<T>
		{
		public:
			T beg; ///< begin value
			T end; ///< end value
			T step; ///< step
			T cur; ///< current value
		public:
			Line(T b, T e, T s) :beg(b), end(e), step(s), cur(b) {}
			virtual void Reset() { cur = beg; }
			virtual bool Next(T &t) 
			{ 
				if ((cur-end>0) == (step>0))
					return false;
				t = cur;
				cur += step;
				return true;
			}
		};
		template <typename T>
		class Ary : public Obj<T>
		{
		public:
			T *p; ///< the array
			int len; ///< the length of the array
			int cur;	///< cur position
		public:
			Ary(T *p_p, int p_len) :p(p_p), len(p_len), cur(0) {}
			virtual void Reset() { cur = 0; }
			virtual bool Next(T &t) 
			{
				if (cur >= len)
					return false;
				t = p[cur];
				cur++;
				return true;
			}
		};
	}
	

	template <typename T>
	class vIter
	{
	public:
		T *m_pBuf; ///< the buffer
		int m_nDim; ///< dimension
		Array<iter::Obj<T>*> m_aObj; ///< the iter of each dimension
		int m_nCur; ///< current number
	public:
		vIter(T *pbuf, int dim) : m_pBuf(pbuf), m_nDim(dim), m_nCur(0) {}
		~vIter(){ SAFE_DEL_POINTER_ARRAY(m_aObj); }
		void Reset()
		{
			for (int i = 0; i < m_nDim; i++) {
				m_aObj[i]->Reset();
			}
			m_nCur = 0;
		}
		bool Next() 
		{
			m_nCur++;
			if (m_nCur == 1) {
				for (int i = 0; i < m_nDim; i++) {
					m_aObj[i]->Next(m_pBuf[i]);
				}
			}
			else {
				int i = 0;
				for (i = 0; i < m_nDim; i++) {
					if (m_aObj[i]->Next(m_pBuf[i]))
						break;
					m_aObj[i]->Reset();
					m_aObj[i]->Next(m_pBuf[i]);
				}
				if (i >= m_nDim) {
					return false;
				}
			}
			
			return true;
		}
		void AddLine(T beg, T end, T step = 1) {
			m_aObj.Add() = new iter::Line<T>(beg, end, step);
		}
		void AddAllLine(T beg, T end, T step = 1) {
			for (int i = 0; i < m_nDim; i++)
				m_aObj[i] = new iter::Line<T>(beg, end, step);
			Reset();
		}
		void AddAry(T *p, int n) {
			m_aObj.Add() = new iter::Ary<T>(p, n);
		}
		void AddAllAry(T *p, int n) {
			for (int i = 0; i < m_nDim; i++)
				m_aObj[i] = new iter::Ary<T>(p, n);
			Reset();
		}
	};
}

#endif

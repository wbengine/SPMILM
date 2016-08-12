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
 * \date 2013-9-23
 * \brief heap, using a linear array. It can be used to sort values.
*/
#ifndef _WB_HEAP_H_
#define _WB_HEAP_H_
#include "wb-vector.h"

namespace wb
{
	/// heap unit
	template <typename TValue, typename TWeight>
	class _wb_HEAP_UNIT_
	{
	public:
		TValue value; ///< value
		TWeight w; ///< weight
		_wb_HEAP_UNIT_(){}
		_wb_HEAP_UNIT_(TValue p_v, TWeight p_w) :value(p_v), w(p_w){}
		///< = opeartion
		void operator = (_wb_HEAP_UNIT_& unit)
		{
			value = unit.value;
			w = unit.w;
		}
	};

	typedef unsigned short _wb_HEAP_MODE_; ///< heap model, can be one of { HEAPMODE_MAXHEAP,HEAPMODE_MINHEAP }
/// max heap, used to sort from large to small
#define HEAPMODE_MAXHEAP	0x0001	
/// min heap, used to sort from samll to large
#define HEAPMODE_MINHEAP	0x0002	

/// heap get father index
#define wbHeap_PARENT(i) ((i)/2)  
/// heap Right index
#define wbHeap_RIGHT(i) (2*(i)+1) 
/// heap Left index
#define wbHeap_LEFT(i) (2*(i))    

	/// heap
	/**
		- make sure the value has the operation =��
		- using Insert to insert data and using OutTop to get data can make the data order.
	*/
	template <typename TValue, typename TWeight>
	class Heap
	{
	private:
		_wb_HEAP_MODE_ m_mode; ///< mode
		Array<_wb_HEAP_UNIT_<TValue, TWeight>> m_aUnits; ///< data��for simplicity, there are no data in [0] position
		_wb_HEAP_UNIT_<TValue, TWeight> *m_pBuf; ///< buffer pointer to the array in m_aUnits

		bool(*m_funpCompare)(TWeight, TWeight); ///< compare function. Compare current node to the father node. Exchage the value if true.

	public:
		Heap(_wb_HEAP_MODE_ mode = HEAPMODE_MAXHEAP, int size = 1) :
			m_mode(mode), m_aUnits(size)
		{
			m_pBuf = m_aUnits.GetBuffer();
			m_aUnits.m_nTop = 0; //������һ�����ݣ���1��ʼ����

			if (mode & HEAPMODE_MAXHEAP)
				m_funpCompare = Heap::MaxHeapCompare;
			else if (mode & HEAPMODE_MINHEAP)
				m_funpCompare = Heap::MinHeapCompare;
		}

		~Heap()
		{

		}

		void Clean() { m_aUnits.Clean(); m_aUnits.m_nTop = 0; }
		int GetNum() { return m_aUnits.GetNum() - 1; }
		bool IsEmpty() { return m_aUnits.m_nTop < 1; }
		/// exchange
		inline void Swap(int i, int j)
		{
			_wb_HEAP_UNIT_<TValue, TWeight> tempUnit;

			//ʹ��m_pBuf��Ѱַ�����ﱣ֤�����ڴ�Խ��
			tempUnit = m_pBuf[i];
			m_pBuf[i] = m_pBuf[j];
			m_pBuf[j] = tempUnit;
		}

		/// update the data at position i and then heapify
		void Update(int i)
		{
			if (i == 1) //���Ǵ�1��ʼʹ�õ�
				return;

			int nParent = wbHeap_PARENT(i);

			//������ǰ�ڵ�͸��׽ڵ�
			if (m_funpCompare(m_pBuf[i].w, m_pBuf[nParent].w))
			{
				Swap(i, nParent);

				Update(nParent);
			}
		}
		/// heapify
		void Heapify(int i)
		{
			if (i > m_aUnits.m_nTop / 2) //��Ҷ�ӽڵ����
				return;

			int nLarge = i;
			int nLeft = wbHeap_LEFT(i);
			int nRight = wbHeap_RIGHT(i);

			if (nLeft <= m_aUnits.m_nTop && m_funpCompare(m_pBuf[nLeft].w, m_pBuf[nLarge].w))
				nLarge = nLeft;
			if (nRight <= m_aUnits.m_nTop && m_funpCompare(m_pBuf[nRight].w, m_pBuf[nLarge].w))
				nLarge = nRight;

			if (nLarge != i)
			{
				//���� i �� nLarge
				Swap(i, nLarge);

				Heapify(nLarge);
			}
		}

		/// insert a value
		void Insert(TValue p_value, TWeight p_w)
		{
			m_aUnits.Add(_wb_HEAP_UNIT_<TValue, TWeight>(p_value, p_w));
			m_pBuf = m_aUnits.GetBuffer();  /// �����������Ԫ�أ��п��ܻ���д�����ڴ棬����bufferָ��

			Update(m_aUnits.m_nTop);
		}

		/// get the value at the top of heap
		bool GetTop(TValue &p_value, TWeight &p_w)
		{
			if (IsEmpty())
				return false;

			p_value = m_aUnits[1].value;
			p_w = m_aUnits[1].w;
			return true;
		}
		/// set the value at top and heapify
		void SetTop(TValue p_value, TWeight p_w)
		{
			m_aUnits[1].value = p_value;
			m_aUnits[1].w = p_w;
			Heapify(1);
		}

		/// out the top
		bool OutTop(TValue &p_value, TWeight &p_w)
		{
			if (IsEmpty())
				return false;

			p_value = m_aUnits[1].value;
			p_w = m_aUnits[1].w;

			m_aUnits[1] = m_aUnits.End();
			m_aUnits.m_nTop--;

			Heapify(1);

			return true;
		}

	private:
		/// compare function for maximum heap
		inline static bool MaxHeapCompare(TWeight p_n, TWeight p_nParent) { return p_n > p_nParent; }
		/// compare function for minimum heap
		inline static bool MinHeapCompare(TWeight p_n, TWeight p_nParent) { return p_n < p_nParent; }
	};
}

#endif

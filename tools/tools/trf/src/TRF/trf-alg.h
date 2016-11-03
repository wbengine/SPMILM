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


#pragma once
#include "trf-def.h"


namespace trf
{

	class Msg;
	class VecIter;
	class Algfb;


	/** 
	 * \class
	 * \brief the definition of a forward-backward template. Perform a forward-backward on the sequence.
	 */
	class Algfb
	{
	public:
		int m_nOrder; ///< the order, i.e. the node number at each cluster {x_1,x_2,...,x_n} 
		int m_nLen; ///< the sequence length.
		int m_nValueLimit; ///< the max-value at each position
		Array<Msg*> m_aAlpha;    ///< the forward message
		Array<Msg*> m_aBeta;	   ///< the backward message

	public:
		Algfb() : m_nOrder(0), m_nLen(0), m_nValueLimit(0) {};
		~Algfb();
		/// prepare 
		void Prepare(int nLen, int nOrder, int nValueLimit);
		/// forward-backward calculation
		void ForwardBackward(int nLen, int nOrder, int nValueLimit);
		/// Get the marginal probability. 'logz' is the input of the log normalization constants
		LogP GetMarginalLogProb(int nPos, int *pSubSeq, int nSubLen, double logz = 0);
		/// Get the summation over the sequence, corresponding to the log normalization constants 'logZ'
		LogP GetLogSummation();
		/// This function need be derived. Calcualte the log probability of each cluster.
		virtual LogP ClusterSum(int *pSeq, int nLen, int nPos, int nOrder) = 0;
	};

	/**
	* \class
	* \brief Forward-backward transfor message
	*/
	class Msg
	{
	private:
		float *m_pbuf; ///< buffer
		int m_dim; ///< the dimension
		int m_size; ///< the size of each dimension

		//Model *m_pmodel; ///< the pointer to the model
	public:
		Msg(int nMsgDim, int nSize);
		~Msg();
		void Fill(float v);
		void Copy(Msg &m);
		float& Get(int *pIdx, int nDim);
		int GetBufSize() const { return pow(m_size, m_dim); }
	};

	/**
	* \class
	* \brief iter all the values in a sub sequence, including iter all the hidden variables and x
	*/
	class VecIter
	{
	public:
		int *m_pBuf; ///< buffer
		int m_nDim;  ///< dimension
		int m_nMin; ///< the min value of each dimension
		int m_nMax; ///< the max value of eahc dimension
	public:
		VecIter(int *p, int nDim, int nMin, int nMax);
		void Reset();
		bool Next();
	};

}
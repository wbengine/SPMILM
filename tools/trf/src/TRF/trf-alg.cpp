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


#include "trf-alg.h"

namespace trf
{
	Algfb::~Algfb()
	{
		for (int i = 0; i < m_aAlpha.GetNum(); i++)
			SAFE_DELETE(m_aAlpha[i]);
		for (int i = 0; i < m_aBeta.GetNum(); i++)
			SAFE_DELETE(m_aBeta[i]);
		m_aAlpha.Clean();
		m_aBeta.Clean();
	}
	void Algfb::Prepare(int nLen, int nOrder, int nValueLimit)
	{
		m_nLen = nLen;
		m_nOrder = nOrder;
		m_nValueLimit = nValueLimit;

		for (int i = 0; i < m_aAlpha.GetNum(); i++)
			SAFE_DELETE(m_aAlpha[i]);
		for (int i = 0; i < m_aBeta.GetNum(); i++)
			SAFE_DELETE(m_aBeta[i]);

		int nClusterNum = nLen - nOrder + 1;
		if (nLen < nOrder)
			return;

		m_aAlpha.SetNum(nClusterNum);
		m_aBeta.SetNum(nClusterNum);

		for (int i = 0; i < nClusterNum; i++) {
			m_aAlpha[i] = new Msg(m_nOrder - 1, m_nValueLimit);
			m_aBeta[i] = new Msg(m_nOrder - 1, m_nValueLimit);
		}
	}
	void Algfb::ForwardBackward(int nLen, int nOrder, int nValueLimit)
	{
		Prepare(nLen, nOrder, nValueLimit);
		int nClusterNum = nLen - nOrder + 1;

		if (nClusterNum <= 0)
			return;

		Vec<int> nodeSeq(nLen); /// the sequence.

		// Forward
		m_aAlpha[0]->Fill(0);
		for (int i = 1; i < nClusterNum; i++) {
			VecIter iter(nodeSeq.GetBuf() + i, m_nOrder - 1, 0, m_nValueLimit - 1);
			while (iter.Next()) {

				LogP dLogSum = LogP_zero;
				for (nodeSeq[i - 1] = 0; nodeSeq[i - 1] < m_nValueLimit; nodeSeq[i - 1]++) {
					LogP temp = ClusterSum(nodeSeq.GetBuf(), nLen, i - 1, m_nOrder) 
						+ m_aAlpha[i - 1]->Get(nodeSeq.GetBuf() + i - 1, m_nOrder - 1);
					dLogSum = Log_Sum(dLogSum, temp);
				}

				m_aAlpha[i]->Get(nodeSeq.GetBuf() + i, m_nOrder - 1) = dLogSum;
			}
		}

		// Backward
		m_aBeta[nClusterNum - 1]->Fill(0);
		for (int i = nClusterNum - 2; i >= 0; i--) {
			VecIter iter(nodeSeq.GetBuf() + i + 1, m_nOrder - 1, 0, m_nValueLimit - 1);
			while (iter.Next()) {

				LogP dLogSum = LogP_zero;
				for (nodeSeq[i + m_nOrder] = 0; nodeSeq[i + m_nOrder] < m_nValueLimit; nodeSeq[i + m_nOrder]++) {
					LogP temp = ClusterSum(nodeSeq.GetBuf(), nLen, i + 1, m_nOrder) 
						+ m_aBeta[i + 1]->Get(nodeSeq.GetBuf() + i + 2, m_nOrder - 1);
					dLogSum = Log_Sum(dLogSum, temp);
				}

				m_aBeta[i]->Get(nodeSeq.GetBuf() + i + 1, m_nOrder - 1) = dLogSum;
			}
		}

	}
	LogP Algfb::GetMarginalLogProb(int nPos, int *pSubSeq, int nSubLen, double logz /* = 0 */)
	{
		// Forward-backward need be calculate

		if (nPos + nSubLen > m_nLen) {
			lout_error("[Model] GetMarginalLogProb: nPos(" << nPos << ")+nOrder(" << nSubLen << ") > seq.len(" << m_nLen << ")!!");
		}

		LogP dSum = LogP_zero;  // 0 prob
		Vec<int> nseq(m_nLen); //save the sequence

		// if the length is very small 
		// then ergodic the sequence of length
		if (m_nLen < m_nOrder)
		{
			VecIter iter(nseq.GetBuf(), m_nLen, 0, m_nValueLimit - 1);
			while (iter.Next()) {
				if (nseq.GetSub(nPos, nSubLen) == VecShell<int>(pSubSeq, nSubLen)) {
					dSum = Log_Sum(dSum, ClusterSum(nseq.GetBuf(), m_nLen, 0, m_nLen));
				}
			}
		}
		else if (nSubLen == m_nOrder) {
			VecShell<int>(nseq.GetBuf()+nPos, nSubLen) = VecShell<int>(pSubSeq, nSubLen);
			dSum = ClusterSum(nseq.GetBuf(), m_nLen, nPos, m_nOrder)
				+ m_aAlpha[nPos]->Get(pSubSeq, m_nOrder - 1)
				+ m_aBeta[nPos]->Get(pSubSeq + 1, m_nOrder - 1);
		}
		else {
			// Choose a cluster
			if (nPos <= m_nLen - m_nOrder) { // choose the cluster nPos
				VecShell<int>(nseq.GetBuf() + nPos, nSubLen) = VecShell<int>(pSubSeq,nSubLen);
				VecIter iter(nseq.GetBuf() + nPos + nSubLen, m_nOrder - nSubLen, 0, m_nValueLimit - 1);
				while (iter.Next()) {
					dSum = Log_Sum(dSum,
						ClusterSum(nseq.GetBuf(), m_nLen, nPos, m_nOrder)
						+ m_aAlpha[nPos]->Get(nseq.GetBuf() + nPos, m_nOrder - 1)
						+ m_aBeta[nPos]->Get(nseq.GetBuf() + nPos + 1, m_nOrder - 1));
				}
			}
			else { // choose the last cluster
				int nCluster = m_nLen - m_nOrder; // cluster position
				VecIter iter(nseq.GetBuf() + nCluster, m_nOrder, 0, m_nValueLimit - 1);
				while (iter.Next()) {
					if (nseq.GetSub(nPos, nSubLen)==VecShell<int>(pSubSeq, nSubLen)) {
						dSum = Log_Sum(dSum,
							ClusterSum(nseq.GetBuf(), m_nLen, nCluster, m_nOrder)
							+ m_aAlpha[nCluster]->Get(nseq.GetBuf() + nCluster, m_nOrder - 1)
							+ m_aBeta[nCluster]->Get(nseq.GetBuf() + nCluster + 1, m_nOrder - 1));
					}
				}
			}
		}

		return dSum - logz;
	}
	LogP Algfb::GetLogSummation()
	{
		int nIterDim = min(m_nOrder, m_nLen);
		Vec<int> nodeSeq(m_nLen);
		int nSumPos = 0;

		/// summation
		LogP logSum = LogP_zero;
		VecIter iter(nodeSeq.GetBuf() + nSumPos, nIterDim, 0, m_nValueLimit - 1);
		while (iter.Next()) {
			LogP temp = 0;
			if (nIterDim == m_nLen) { // no cluster
				temp = ClusterSum(nodeSeq.GetBuf(), m_nLen, nSumPos, nIterDim);
			}
			else {
				temp = ClusterSum(nodeSeq.GetBuf(), m_nLen, nSumPos, m_nOrder)
					+ m_aAlpha[nSumPos]->Get(nodeSeq.GetBuf() + nSumPos, m_nOrder - 1)
					+ m_aBeta[nSumPos]->Get(nodeSeq.GetBuf() + nSumPos + 1, m_nOrder - 1);
			}

			logSum = Log_Sum(logSum, temp);
		}

		return logSum;
	}

	/************************************************************************/
	/*  class Msg                                                                     */
	/************************************************************************/

	Msg::Msg(int nMsgDim, int nSize)
	{
		m_dim = nMsgDim;
		//m_pmodel = pm;
		m_size = nSize;//m_pmodel->GetEncodeNodeLimit();
		int totalsize = pow(m_size, m_dim);
		m_pbuf = new float[totalsize];

		Fill(0);
	}
	Msg::~Msg()
	{
		SAFE_DELETE_ARRAY(m_pbuf);
	}
	void Msg::Fill(float v)
	{
		int totalsize = pow(m_size, m_dim);
		for (int i = 0; i < totalsize; i++) {
			m_pbuf[i] = v;
		}
	}
	void Msg::Copy(Msg &m)
	{
		if (GetBufSize() != m.GetBufSize()) {
			SAFE_DELETE_ARRAY(m_pbuf);
			m_dim = m.m_dim;
			m_size = m.m_size;
			m_pbuf = new float[GetBufSize()];
		}

		memcpy(m_pbuf, m.m_pbuf, sizeof(m_pbuf[0]) * GetBufSize());
	}
	float& Msg::Get(int *pIdx, int nDim)
	{
		lout_assert(nDim == m_dim);

		int nIndex = pIdx[0];
		for (int i = 0; i < nDim - 1; i++)
		{
			nIndex = nIndex * m_size + pIdx[i + 1];
		}
		return m_pbuf[nIndex];
	}



	/************************************************************************/
	/* VecIter                                                              */
	/************************************************************************/
	VecIter::VecIter(int *p, int nDim, int nMin, int nMax)
	{
		m_pBuf = p;
		m_nDim = nDim;
		m_nMin = nMin;
		m_nMax = nMax;
		Reset();
	}
	void VecIter::Reset()
	{
		for (int i = 0; i < m_nDim; i++)
			m_pBuf[i] = m_nMin;
		m_pBuf[0]--;
	}
	bool VecIter::Next()
	{
		m_pBuf[0]++;
		for (int i = 0; i < m_nDim - 1; i++) {
			if (m_pBuf[i] > m_nMax) {
				m_pBuf[i + 1]++;
				m_pBuf[i] = m_nMin;
			}
			else {
				break;
			}
		}

		return m_pBuf[m_nDim - 1] <= m_nMax;
	}
}
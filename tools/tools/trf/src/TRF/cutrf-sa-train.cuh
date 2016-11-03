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

#ifndef _CUTRF_SA_TRAIN_CUH_
#define _CUTRF_SA_TRAIN_CUH_

#include "cutrf-model.cuh"
#include "trf-sa-train.h"

namespace trf
{
	class cuSAFunc : public trf::SAfunc
	{
	public:
		cutrf::Model dev_m; ///< the model on GPU
		cu::Array<cutrf::Seq> dev_aSeq; ///< store the sampled sequence of each thread

		cu::Mat<int> dev_matFeatFind;
	public:
		cuSAFunc() {}
		/// reset 
		virtual void Reset(trf::Model *pModel, 
			trf::CorpusBase *pTrain, trf::CorpusBase *pValid = NULL, trf::CorpusBase *pTest = NULL, 
			int nMinibatch = 100)
		{
			trf::SAfunc::Reset(pModel, pTrain, pValid, pTest, nMinibatch);


			dev_m.Copy(*pModel, nMinibatch);

			trf::Seq seq;
			wb::Array<cutrf::Seq> aSeq;
			for (int i = 0; i < nMinibatch; i++) {
				int nLen = LineSampling(m_samplePi.GetBuf(), m_samplePi.GetSize());
				lout_assert(nLen > 0 && nLen <= m_pModel->GetMaxLen());
				//int nLen = 5;
				seq.Reset(nLen);
				seq.Random(pModel->GetVocab());
				aSeq[i].Init(pModel->GetMaxLen() + 10);
				aSeq[i].Copy(seq);
			}
			dev_aSeq.Copy(aSeq);

			dev_matFeatFind.Reset(nMinibatch, FEAT_FIND_MAX);

			//Pause();
		}
		/// calcualte the expectation of SA samples
		virtual void GetSampleExp(VecShell<double> &vExp, VecShell<double> &vLen)
		{
			wb::Clock clk;
			clk.Begin();
			cudaSample(dev_matFeatFind, dev_aSeq, dev_m, ceil(1.0*m_nMiniBatchSample/128), 128);
			lout << "cudaSampleTimes=" << clk.ToSecond(clk.Get()) << "s" << endl;


			wb::Mat<int> matFeatFind;
			dev_matFeatFind.CopyTo(matFeatFind);
			lout << "matCopyBackTimes=" << clk.ToSecond(clk.Get()) << "s" << endl;

			vExp.Fill(0);
			vLen.Fill(0);

			for (int t = 0; t < m_nMiniBatchSample; t++) {
				VecShell<int> v = matFeatFind[t];
				int nLen = v[0];
				for (int i = 1; i < v.GetSize() && v[i] != VocabID_none; i++) {
					vExp[v[i]] += m_trainPi[nLen] / m_pModel->m_pi[nLen];
				}
				vLen[nLen] += 1;
			}
			lout << "SumCountTimes=" << clk.ToSecond(clk.Get()) << "s" << endl;

			

// 			trf::Seq seq;
// 			wb::Array<cutrf::Seq> aSeq;
// 			dev_aSeq.CopyTo(aSeq);
// 
// 			for (int i = 0; i < aSeq.GetNum(); i++) {
// 				aSeq[i].CopyTo(seq);
// 				int nLen = seq.GetLen();
// 				m_pModel->FeatCount(seq, vExp.GetBuf(), m_trainPi[nLen] / m_pModel->m_pi[nLen]);
// 				vLen[nLen]++;
// 
// 				if (m_fsamp) {
// 					seq.Print(m_fsamp);
// 				}
// 			}

			m_vAllSampleLenCount += vLen; /// save the length count
			m_vCurSampleLenCount.Copy(vLen); /// save current length count
			m_nTotalSample += m_nMiniBatchSample;

			vExp /= m_nMiniBatchSample;
			vLen /= m_nMiniBatchSample;

			lout << "SampleEnd=" << clk.ToSecond(clk.Get()) << "s" << endl;
		}

// 		virtual void GetGradient(double *pdGradient)
// 		{
// 			int nWeightNum = m_pModel->GetParamNum();
// 			m_vSampleExp.Reset(nWeightNum);
// 			m_vSampleLen.Reset(m_pModel->GetMaxLen() + 1);
// 
// 
// 			/* get theoretical expectation */
// 			GetSampleExp(m_vSampleExp, m_vSampleLen);
// 
// 
// 
// 			/* Calculate the gradient */
// 			for (int i = 0; i < nWeightNum; i++) {
// 				pdGradient[i] = (
// 					m_vEmpiricalExp[i] - m_vSampleExp[i]
// 					- m_fRegL2 * m_pModel->m_value[i] // the L2 regularization
// 					);
// 			}
// 
// 			/*
// 			Zeta update
// 			*/
// 			for (int l = 0; l <= m_pModel->GetMaxLen(); l++) {
// 				if (m_pModel->m_pi[l] > 0) {
// 					pdGradient[nWeightNum + l] = min(10.0, m_vSampleLen[l] / m_pModel->m_pi[l]);
// 				}
// 				else {
// 					pdGradient[nWeightNum + l] = 0;
// 				}
// 			}
// 
// 
// 			if (m_fgrad.Good()) {
// 				m_fgrad.PrintArray("%f ", pdGradient, m_nParamNum);
// 				m_fgrad.Print("\n");
// 			}
// 			if (m_fexp.Good()) {
// 				m_fexp.PrintArray("%f ", m_vSampleExp.GetBuf(), m_vSampleExp.GetSize());
// 				m_fexp.Print("\n");
// 			}
// 
// 		}
		/// Revise the paremeters of dev_m
		virtual void SetParam(double *pdParams)
		{
			trf::SAfunc::SetParam(pdParams);
			wb::Clock clk;
			clk.Begin();
			lout << "Copy Parameters to Device...";
			dev_m.m_value.Copy(m_pModel->m_value);
			dev_m.m_zeta.Copy(m_pModel->m_zeta);
			dev_m.m_logz.Copy(m_pModel->m_logz);
			lout << "\r" << "Copy Parameters to Device [time=" << clk.ToSecond(clk.End()) << "]" << endl;
		}

		
	};
}


#endif
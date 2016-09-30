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


#include "trf-sa-train.h"
#include "wb-log.h"

namespace trf
{
	ThreadData::~ThreadData()
	{
		for (int i = 0; i < aSeqs.GetNum(); i++) {
			SAFE_DELETE(aSeqs[i]);
		}
	}
	void ThreadData::Create(int maxlen, Model *pModel)
	{
		aSeqs.SetNum(maxlen + 1);
		aSeqs.Fill(NULL);
		for (int i = 1; i < aSeqs.GetNum(); i++) {
			aSeqs[i] = new Seq(i);
			aSeqs[i]->Random(pModel->m_pVocab);
		}
	}

	void SAfunc::Reset(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid /* = NULL */, CorpusBase *pTest /* = NULL */, int nMinibatch /* = 100 */)
	{
		MLfunc::Reset(pModel, pTrain, pValid, pTest);
		GetEmpVar(pTrain, m_vEmpiricalVar);

		m_nMiniBatchSample = nMinibatch;

		/*
		sampling pi
		*/
		lout << "Smoothing the pi" << endl;
		double dMax = 0;
		int iMax = 0;
		for (int i = 1; i < m_trainPi.GetSize(); i++) {
			if (m_trainPi[i] > dMax) {
				dMax = m_trainPi[i];
				iMax = i;
			}
		}
		m_samplePi.Copy(m_trainPi);
		for (int i = 1; i < iMax; i++) {
			m_samplePi[i] = dMax;
		}
		for (int i = 1; i < m_samplePi.GetSize(); i++) {
			m_samplePi[i] = max((double)m_samplePi[i], 1e-5);
		}
		LineNormalize(m_samplePi.GetBuf() + 1, m_samplePi.GetSize() - 1);

		lout << "sample-pi = [ "; lout.output(m_samplePi.GetBuf() + 1, m_samplePi.GetSize() - 1); lout << "]" << endl;
		m_pModel->SetPi(m_samplePi.GetBuf());

		/* save the sample count */
		m_vAllSampleLenCount.Reset(m_pModel->GetMaxLen() + 1);
		m_vCurSampleLenCount.Reset(m_pModel->GetMaxLen() + 1);
		m_vAllSampleLenCount.Fill(0);
		m_nTotalSample = 0;

		/* for SA estimateio. there are two set of paremeters
		    i.e. feature weight \lambda and normalization constants \zeta
		*/
		m_nParamNum = m_pModel->GetParamNum() + m_pModel->GetMaxLen() + 1;

		m_nCDSampleTimes = 1;
		m_nSASampleTimes = 1;
		
	}
	void SAfunc::PrintInfo()
	{
		lout << "[SAfunc] *** Info: *** " << endl;
		lout << "  "; lout_variable(m_nMiniBatchSample);
		lout << "  "; lout_variable(m_var_gap);
		lout << "  "; lout_variable(m_fRegL2);
		lout << "[SAfunc] *** [End] ***" << endl;
	}
	void SAfunc::RandSeq(Seq &seq, int nLen /* = -1 */)
	{
		if (nLen == -1) {
			nLen = rand() % m_pModel->GetMaxLen() + 1; /// [1, maxlen]
		}
		
		seq.Reset(nLen);
		seq.Random(m_pModel->m_pVocab);
	}
	void SAfunc::SetParam(double *pdParams)
	{
		if (pdParams == NULL)
			return;

		m_value.Reset(m_pModel->GetParamNum());
		for (int i = 0; i < m_value.GetSize(); i++)
			m_value[i] = (PValue)pdParams[i];
		m_pModel->SetParam(m_value.GetBuf());
		m_pModel->ExactNormalize(1); // only calculate Z_1

		/* set zeta */
		m_pModel->SetZeta(pdParams + m_pModel->GetParamNum());

		if (m_fparm.Good()) {
			m_fparm.PrintArray("%f ", pdParams, m_nParamNum);
		}
	}
	void SAfunc::GetParam(double *pdParams)
	{
		if (pdParams == NULL)
			return;

		/* get lambda */
		m_value.Reset(m_pModel->GetParamNum());
		m_pModel->GetParam(m_value.GetBuf());
		for (int i = 0; i < m_value.GetSize(); i++)
			pdParams[i] = m_value[i];
		/* get zeta */
		pdParams += m_pModel->GetParamNum();
		for (int i = 0; i <= m_pModel->GetMaxLen(); i++) {
			pdParams[i] = m_pModel->m_zeta[i];
		}
	}

	int qsort_compare_double(const void * a, const void * b)
	{
		if (*(double*)a < *(double*)b) return -1;
		if (*(double*)a == *(double*)b) return 0;
		if (*(double*)a > *(double*)b) return 1;
	}

	void SAfunc::GetEmpVar(CorpusBase *pCorpus, Vec<double> &vVar)
	{
		int nThread = omp_get_max_threads();
		
		// the true length distribution
		Prob *pi = m_trainPi.GetBuf();

		vVar.Fill(0);
		Array<VocabID> aSeq;
		Vec<double> vExpf2(m_pModel->GetParamNum());
		Vec<double> vExp_l(m_pModel->GetParamNum());

		Mat<double> matExpf2(nThread, vExpf2.GetSize());
		Mat<double> matExp_l(nThread, vExp_l.GetSize());

		vExpf2.Fill(0);
		vExp_l.Fill(0);
		matExpf2.Fill(0);
		matExp_l.Fill(0);

		/// Count p[f^2]
		lout.Progress(0, true, pCorpus->GetNum() - 1, "[SAfunc] E[f^2]:");
#pragma omp parallel for firstprivate(aSeq)
		for (int l = 0; l < pCorpus->GetNum(); l++) {

			double *pExpf2 = matExpf2[omp_get_thread_num()].GetBuf();

			pCorpus->GetSeq(l, aSeq);
			Seq seq;
			seq.Set(aSeq, m_pModel->m_pVocab);
			
			int nLen = min(m_pModel->GetMaxLen(), seq.GetLen());

			LHash<int, int> aFeatNum;
			bool bFound;
			Array<int> afeat;
			m_pModel->m_pFeat->Find(afeat, seq);
			for (int i = 0; i < afeat.GetNum(); i++) {
				int *p = aFeatNum.Insert(afeat[i], bFound);
				if (!bFound) *p = 0;
				(*p) += 1;
			}
			LHashIter<int, int> iter(&aFeatNum);
			int *pCount;
			int nFeat;
			while (pCount = iter.Next(nFeat)) {
				pExpf2[nFeat] += pow((double)(*pCount), 2);
			}
#pragma omp critical 
			{
				lout.Progress();
			}
		}

		vExpf2.Fill(0);
		for (int t = 0; t < nThread; t++) {
			vExpf2 += matExpf2[t];
		}
		vExpf2 /= pCorpus->GetNum();


		//lout_variable(aExpFeatSqu[38272]);

		/// Count p_l[f]
		/// As save p_l[f] for all the length cost too much memory. So we calculate each p_l[f] separately.
		lout.Progress(0, true, m_pModel->GetMaxLen(), "[SAfunc] E_l[f]:");
		for (int nLen = 1; nLen <= m_pModel->GetMaxLen(); nLen++)
		{
			matExp_l.Fill(0);
			
			Array<int> aSeqId;
			/// find all the sequence with length nLen
			for (int i = 0; i < pCorpus->GetNum(); i++) {
				pCorpus->GetSeq(i, aSeq);
				int nSeqLen = aSeq.GetNum();
				if (nLen == m_pModel->GetMaxLen()) {
					if (nSeqLen < nLen)
						continue;
				}
				else {
					if (nSeqLen != nLen)
						continue;
				}
				aSeqId.Add(i);
			}

#pragma omp parallel for firstprivate(aSeq)
			for (int k = 0; k < aSeqId.GetNum(); k++) 
			{
				pCorpus->GetSeq(aSeqId[k], aSeq);

				Seq seq;
				seq.Set(aSeq, m_pModel->m_pVocab);
				m_pModel->FeatCount(seq, matExp_l[omp_get_thread_num()].GetBuf());
			}

			if (aSeqId.GetNum() > 0) {
				vExp_l.Fill(0);
				for (int t = 0; t < nThread; t++) {
					vExp_l += matExp_l[t];
				}
				vExp_l /= aSeqId.GetNum();
			}
			else {
				vExp_l.Fill(0);
			}


			for (int i = 0; i < m_pModel->GetParamNum(); i++) 
				vExpf2[i] -= pi[nLen] * pow(vExp_l[i], 2);  /// calcualte p[f^2] - \pi_l * p_l[f]^2

			lout.Progress(nLen);
		}

		/// output the zero number
		int nZero = 0;
		int nDownGap = 0;
		double dMinVarOverZero = 100;
		for (int i = 0; i < m_nParamNum; i++) {
			if (vExpf2[i] == 0)
				nZero++;
			else
				dMinVarOverZero = min(vExpf2[i], dMinVarOverZero);

			if (vExpf2[i] < m_var_gap) {
				nDownGap++;
				vExpf2[i] = m_var_gap;
			}
				
		}
		if (nZero > 0) {
			lout_warning("[EmpiricalVar] Exist zero expectation  (zero-num=" << nZero << ")");
		}
		lout << "[EmpiricalVar] the number of ( var < gap ) is " << nDownGap << endl;
		lout << "[EmpiricalVar] min variance value (over 0) is " << dMinVarOverZero << endl;


		///save
		vVar.Copy(vExpf2);

		// Write
		if (m_fmean.Good()) {
			lout << "Write Empirical Mean ..." << endl;
			Vec<PValue> aLogExp(m_vEmpiricalExp.GetSize());
			for (int i = 0; i < aLogExp.GetSize(); i++) aLogExp[i] = log(m_vEmpiricalExp[i]);
			m_pModel->m_pFeat->WriteT(m_fmean, aLogExp.GetBuf());
//			m_fmean.PrintArray("%f\n", m_vEmpiricalExp.GetBuf(), m_vEmpiricalExp.GetSize());
		}
		if (m_fvar.Good()) {
			lout << "Write Empirical Var ..." << endl;
			Vec<PValue> aLogVar(vVar.GetSize());
			for (int i = 0; i < vVar.GetSize(); i++) aLogVar[i] = log(vVar[i]);
			m_pModel->m_pFeat->WriteT(m_fvar, aLogVar.GetBuf());
			//m_fvar.PrintArray("%f\n", vVar.GetBuf(), vVar.GetSize());
		}
	}

	void SAfunc::GetSampleExp(VecShell<double> &vExp, VecShell<double> &vLen)
	{
		int nThread = omp_get_max_threads();
		m_matSampleExp.Reset(nThread, m_pModel->GetParamNum());
		m_matSampleLen.Reset(nThread, m_pModel->GetMaxLen() + 1);
//		Vec<int> vNum(nThread); // record the sample number of each thread

		m_matSampleExp.Fill(0);
		m_matSampleLen.Fill(0);
//		vNum.Fill(0);


		// init the sequence
		if (m_threadSeq.GetNum() != nThread) {
			for (int i = 0; i < nThread; i++) {
				m_threadSeq[i] = new Seq;
				RandSeq(*m_threadSeq[i]);
			}
		}

		/* sampling */
		//lout.Progress(0, true, m_nMiniBatchSample-1, "[SA] sample:");
#pragma omp parallel for
		for (int sample = 0; sample < m_nMiniBatchSample; sample++)
		{
			int tid = omp_get_thread_num();
			m_pModel->Sample(*m_threadSeq[tid]);

			int nLen = min(m_pModel->GetMaxLen(), m_threadSeq[tid]->GetLen());

			//m_aSeqs[threadID]->Print();
			m_pModel->FeatCount(*m_threadSeq[tid], m_matSampleExp[tid].GetBuf(), m_trainPi[nLen] / m_pModel->m_pi[nLen]);
			m_matSampleLen[tid][nLen]++;
			//			vNum[threadID]++;

#pragma omp critical
			{
				if (m_fsamp.Good()) {
					m_threadSeq[tid]->Print(m_fsamp);
				}
				//lout.Progress();
			}
			
		}
		lout << " len-jump acc-rate=";
		lout_variable_rate(m_pModel->m_nLenJumpAccTimes, m_pModel->m_nLenJumpTotalTime);
		m_pModel->m_nLenJumpAccTimes = 0;
		m_pModel->m_nLenJumpTotalTime = 0;
		lout << " class-propose acc-rate=";
		lout_variable_rate(m_pModel->m_nSampleHAccTimes, m_pModel->m_nSampleHTotalTimes);
		m_pModel->m_nSampleHAccTimes = 0;
		m_pModel->m_nSampleHTotalTimes = 0;
		lout << endl;



		// summarization
		vExp.Fill(0);
		vLen.Fill(0);
		for (int t = 0; t < nThread; t++) {
			vExp += m_matSampleExp[t];
			vLen += m_matSampleLen[t];
		}
		m_vAllSampleLenCount += vLen; /// save the length count
		m_vCurSampleLenCount.Copy(vLen); /// save current length count
		m_nTotalSample += m_nMiniBatchSample;

		vExp /= m_nMiniBatchSample;
		vLen /= m_nMiniBatchSample;
	}
	
	void SAfunc::IterEnd(double *pFinalParams)
	{
		SetParam(pFinalParams);
		// set the pi as the len-prob in training set.
		m_pModel->SetPi(m_trainPi.GetBuf());
	}
	void SAfunc::WriteModel(int nEpoch)
	{
		String strTempModel;
		String strName = String(m_pathOutputModel).FileName();
#ifdef __linux
		strTempModel.Format("%s.n%d.model", strName.GetBuffer(), nEpoch);
#else
		strTempModel.Format("%s.n%d.model", strName.GetBuffer(), nEpoch);
#endif
		// set the pi as the pi of training set
		m_pModel->SetPi(m_trainPi.GetBuf());
		m_pModel->WriteT(strTempModel);
		m_pModel->SetPi(m_samplePi.GetBuf());
	}
	void SAfunc::GetGradient(double *pdGradient)
	{
		int nWeightNum = m_pModel->GetParamNum();
		m_vSampleExp.Reset(nWeightNum);
		m_vSampleLen.Reset(m_pModel->GetMaxLen() + 1);

		
// 		/* get theoretical expectation */
 		GetSampleExp(m_vSampleExp, m_vSampleLen);

		

		/* Calculate the gradient */
		for (int i = 0; i < nWeightNum; i++) {
			pdGradient[i] = (
				m_vEmpiricalExp[i] - m_vSampleExp[i]
				- m_fRegL2 * m_pModel->m_value[i] // the L2 regularization
				) / ( m_vEmpiricalVar[i] + m_fRegL2 ) /**( m_fEmpiricalVarGap + m_fRegL2 )*/; // rescaled by variance
		}

		/* dropout operation */
// 		lout << "dropout operation..." << endl;
// 		double m_dDropoutRate = 0.2;
// 		Array<int> aIndex(nWeightNum);
// 		for (int i = 0; i < nWeightNum; i++)
// 			aIndex[i] = i;
// 
// 		RandomPos(aIndex.GetBuffer(), nWeightNum, m_dDropoutRate*nWeightNum);
// 		for (int i = 0; i < m_dDropoutRate*nWeightNum; i++) {
// 			pdGradient[aIndex[i]] = 0;
// 		}
// 
// 		int nZero = 0;
// 		for (int i = 0; i < nWeightNum; i++) {
// 			if (pdGradient[i] == 0) nZero++;
// 		}
// 		lout_variable_precent(nZero, nWeightNum);

		/*
			Zeta update
		*/
		for (int l = 0; l <= m_pModel->GetMaxLen(); l++) {
			if (m_pModel->m_pi[l] > 0) {
				pdGradient[nWeightNum + l] =  m_vSampleLen[l] / m_pModel->m_pi[l];
			}	
			else {
				pdGradient[nWeightNum + l] = 0;
			}
		}

		
		if (m_fgrad.Good()) {
			m_fgrad.PrintArray("%f ", pdGradient, m_nParamNum);
			m_fgrad.Print("\n");
		}
		if (m_fexp.Good()) {
			m_fexp.PrintArray("%f ", m_vSampleExp.GetBuf(), m_vSampleExp.GetSize());
			m_fexp.Print("\n");
		}
		


	}
	int SAfunc::GetExtraValues(int t, double *pdValues)
	{
		int nValue = 0;

		// set the training pi
		m_pModel->SetPi(m_trainPi.GetBuf());

		Vec<Prob> samsZeta(m_pModel->m_zeta.GetSize());
		Vec<Prob> trueZeta(m_pModel->m_zeta.GetSize());
		//Vec<double> trueLogZ(m_pModel->m_logz.GetSize()); 
		samsZeta.Fill(0);
		trueZeta.Fill(0);
		samsZeta = m_pModel->m_zeta;

		/* AIS to calcualte the p(v) */
		if (m_pCorpusTrain) pdValues[nValue++] = -GetLL(m_pCorpusTrain);
		if (m_pCorpusValid) pdValues[nValue++] = -GetLL(m_pCorpusValid);
 		if (m_pCorpusTest) pdValues[nValue++] = -GetLL(m_pCorpusTest);

		/* true Z_L to get the LL */
		if (m_pModel->m_pVocab->GetSize() < 100 && m_pModel->GetMaxOrder() < 4) {

			m_pModel->ExactNormalize(); // normalization
			trueZeta.Copy(m_pModel->m_zeta);
			if (m_pCorpusTrain) pdValues[nValue++] = -GetLL(m_pCorpusTrain);
			if (m_pCorpusValid) pdValues[nValue++] = -GetLL(m_pCorpusValid);
			if (m_pCorpusTest) pdValues[nValue++] = -GetLL(m_pCorpusTest);

			m_pModel->SetZeta(samsZeta.GetBuf());	
		}


		/* output debug */
		if (!m_fdbg.Good()) {
			m_fdbg.Open("SAfunc.dbg", "wt");
		}
		m_vAllSampleLenCount *= 1.0 / m_nTotalSample;
		m_vCurSampleLenCount *= 1.0 / m_nMiniBatchSample;
		m_fdbg.PrintArray("%f ", m_vCurSampleLenCount.GetBuf() + 1, m_vCurSampleLenCount.GetSize() - 1);
		m_fdbg.PrintArray("%f ", m_vAllSampleLenCount.GetBuf() + 1, m_vAllSampleLenCount.GetSize() - 1);
		m_fdbg.PrintArray("%f ", m_samplePi.GetBuf() + 1, m_samplePi.GetSize() - 1);
		m_fdbg.PrintArray("%f ", trueZeta.GetBuf() + 1, trueZeta.GetSize() - 1);
		m_fdbg.PrintArray("%f ", samsZeta.GetBuf() + 1, samsZeta.GetSize() - 1);
		m_fdbg.Print("\n");
		m_vAllSampleLenCount *= m_nTotalSample;
		m_vCurSampleLenCount *= m_nMiniBatchSample;

		m_pModel->SetPi(m_samplePi.GetBuf());

		return nValue;
	}

	void LearningRate::Reset(const char *pstr, int p_t0)
	{
		sscanf(pstr, "%lf,%lf", &tc, &beta);
		t0 = p_t0;
		//lout << "[Learning Rate] tc=" << tc << " beta=" << beta << " t0=" << t0 << endl;
	}
	double LearningRate::Get(int t)
	{
		double gamma;
		if (t <= t0) {
			gamma = 1.0 / (tc + pow(t, beta));
		}
		else {
			gamma = 1.0 / (tc + pow(t0, beta) + t - t0);
		}
		return gamma;
	}


	bool SAtrain::Run(const double *pInitParams /* = NULL */)
	{
		if (!m_pfunc) {
			lout_Solve << "m_pFunc == NULL" << endl;
			return false;
		}
		Clock ck;
		m_dSpendMinute = 0;

		SAfunc *pSA = (SAfunc*)m_pfunc;
// 		int nIterPerEpoch = pSA->m_pCorpusTrain->GetNum() / pSA->m_nMiniBatchSample + 1;
// 		lout_variable(nIterPerEpoch);

		
		double *pdCurParams = new double[m_pfunc->GetParamNum()]; 
		double *pdCurGradient = new double[m_pfunc->GetParamNum()];
		double *pdCurDir = new double[m_pfunc->GetParamNum()]; // current update direction
 		double dCurValue = 0; 
		double dExValues[Func::cn_exvalue_max_num]; 
		double nExValueNum;

		// if average
		bool bAvg = (m_nAvgBeg > 0);
		double *pdAvgParams = NULL;
		if (bAvg) {
			pdAvgParams = new double[m_pfunc->GetParamNum()];
		}


		
		for (int i = 0; i < m_pfunc->GetParamNum(); i++) {
			pdCurParams[i] = (pInitParams) ? pInitParams[i] : 1;
		}
		memset(pdCurGradient, 0, sizeof(double)*m_pfunc->GetParamNum());
		memset(pdCurDir, 0, sizeof(double)*m_pfunc->GetParamNum());

		IterInit(); ///init
		m_pfunc->SetParam(pdCurParams);
		//pSA->WriteModel(0);

		// iteration begin
		lout_Solve << "************* Training Begin *****************" << endl;
		lout_Solve << "print-per-iter=" << m_nPrintPerIter << endl;
		lout.bOutputCmd() = false;
		ck.Begin();
		for (m_nIterNum = m_nIterMin; m_nIterNum <= m_nIterMax; m_nIterNum++)
		{
			// epoch number
			m_fEpochNun = 1.0 * m_nIterNum * pSA->m_nMiniBatchSample / pSA->m_pCorpusTrain->GetNum();

			// set the parameter
			m_pfunc->SetParam(pdCurParams);
			// get the gradient
			m_pfunc->GetGradient(pdCurGradient);
			// get the function value
			dCurValue = m_pfunc->GetValue();
			// get the averaged parameters
			if (bAvg) {
				if (m_nIterNum <= m_nAvgBeg) {
					memcpy(pdAvgParams, pdCurParams, sizeof(pdCurParams[0])*m_pfunc->GetParamNum());
				}
				else {
					for (int i = 0; i < m_pfunc->GetParamNum(); i++) {
						pdAvgParams[i] += (pdCurParams[i] - pdAvgParams[i]) / (m_nIterNum - m_nAvgBeg);
					}
				}
			}

			// print
			if (m_nIterNum % m_nPrintPerIter == 0 || m_nIterNum == m_nIterMax)
			{
				m_dSpendMinute = ck.ToSecond(ck.Get()) / 60;
				bool bPrintCmd;

				bPrintCmd = lout.bOutputCmd();
				lout.bOutputCmd() = true;
				lout_Solve << "t=" << m_nIterNum;
				cout<<setprecision(4)<<setiosflags(ios::fixed);
				lout << " epoch=" << m_fEpochNun;
				cout<<setprecision(2)<<setiosflags(ios::fixed);
				lout << " time="  << m_dSpendMinute << "m";
				lout << (bAvg ? " [Avg]" : " ");
				lout.bOutputCmd() = bPrintCmd;


				// get the ex-value
				if (bAvg) pSA->SetParam(pdAvgParams); ///< set average
				// This function will use AIS to normaliza the model
				nExValueNum = pSA->GetExtraValues(m_nIterNum, dExValues);
				
				bPrintCmd = lout.bOutputCmd();
				lout.bOutputCmd() = true;
				lout<< "ExValues={ ";
				cout<< setprecision(2) << setiosflags(ios::fixed);
				for (int i = 0; i < nExValueNum; i++)
					lout << dExValues[i] << " ";
				lout << "}" << endl;
				
				// write model
				if (m_aWriteAtIter.Find(m_nIterNum) != -1)
					pSA->WriteModel(m_nIterNum);

				lout.bOutputCmd() = bPrintCmd;

				if (bAvg) pSA->SetParam(pdCurParams); ///< set back
			}
			//lout.Progress(m_nIterNum % m_nPrintPerIter, true, m_nPrintPerIter - 1, "Train:");

				
				

			/* Stop Decision */
			if (StopDecision(m_nIterNum, dCurValue, pdCurGradient)) {
				break;
			}


			// update the learning rate gamma
			UpdateGamma(m_nIterNum);

			// update the direction
			UpdateDir(pdCurDir, pdCurGradient, pdCurParams);

			// Update parameters
			Update(pdCurParams, pdCurDir, 0);
		}

		lout_Solve << "************* Training End *****************" << endl;
		lout_Solve << "iter=" << m_nIterNum << " time=" << m_dSpendMinute << "m" << endl;
		lout_Solve << "********************************************" << endl;

		// do something at the end of the iteration
		if (bAvg) pSA->IterEnd(pdAvgParams);
		else pSA->IterEnd(pdCurParams);

		SAFE_DELETE_ARRAY(pdCurGradient);
		SAFE_DELETE_ARRAY(pdCurDir);
		SAFE_DELETE_ARRAY(pdCurParams);
		SAFE_DELETE_ARRAY(pdAvgParams);
		return true;
	}

	void SAtrain::UpdateGamma(int nIterNum)
	{
		m_gamma_lambda = m_gain_lambda.Get(nIterNum);
		m_gamma_zeta = m_gain_zeta.Get(nIterNum);
		
		lout_Solve << "g_lambda=" << m_gamma_lambda
			<< " g_zeta=" << m_gamma_zeta
			<< endl;
	}
	void SAtrain::UpdateDir(double *pDir, double *pGradient, const double *pdParam)
	{
		/* using the momentum */
		// pdDir is actually the gradient

		SAfunc* pSA = (SAfunc*)m_pfunc;
		int nWeightNum = pSA->GetFeatNum();
		int nZetaNum = pSA->GetZetaNum();

		lout_assert(nWeightNum + nZetaNum == m_pfunc->GetParamNum());

		// update lambda
		for (int i = 0; i < nWeightNum; i++) {
			pDir[i] = m_gamma_lambda * pGradient[i];
		}

		if (m_dir_gap > 0) {
			int n_dgap_cutnum = 0;
			for (int i = 0; i < nWeightNum; i++) {
				if (pDir[i] > m_dir_gap) {
					pDir[i] = m_dir_gap;
					n_dgap_cutnum++;
				}
				else if (pDir[i] < -m_dir_gap) {
					pDir[i] = -m_dir_gap;
					n_dgap_cutnum++;
				}
			}
			lout_variable_precent(n_dgap_cutnum, nWeightNum);
		}
		

		// update zeta
		for (int i = nWeightNum; i < nWeightNum + nZetaNum; i++) {
			// limit the update of zeta.
			pDir[i] = min( m_gamma_zeta, 1.0*pSA->m_pModel->GetMaxLen()*pSA->m_pModel->m_pi[i-nWeightNum] ) * pGradient[i];
		}
	
	}
	void SAtrain::Update(double *pdParam, const double *pdDir, double dStep)
	{
		// pdDir is actually the gradient

		SAfunc* pSA = (SAfunc*)m_pfunc;
		int nWeightNum = pSA->GetFeatNum();
		int nZetaNum = pSA->GetZetaNum();

//		lout_assert(nWeightNum == nNgramFeatNum + nVHsize + nCHsize + nHHsize);

		// update lambda
		if (m_bUpdate_lambda) {
			for (int i = 0; i < nWeightNum; i++) {
				pdParam[i] += pdDir[i];
			}
		}

		

		// update zeta
		if (m_bUpdate_zeta) {
			for (int i = nWeightNum; i < nWeightNum + nZetaNum; i++) {
				pdParam[i] += pdDir[i];
			}
			double zeta1 = pdParam[nWeightNum + 1];
			for (int i = nWeightNum + 1; i < nWeightNum + nZetaNum; i++) {
				pdParam[i] -= zeta1; // minus the zeta[1];
			}
		}

		
	}

#define GAIN_INFO(g) lout<<"  "#g"\ttc="<<g.tc<<" beta="<<g.beta<<" t0="<<g.t0<<endl;
	void SAtrain::PrintInfo()
	{
		lout << "[SATrain] *** Info: ***" << endl;
		GAIN_INFO(m_gain_lambda);
		GAIN_INFO(m_gain_zeta);
		lout << "  " << "m_dir_gap=" << m_dir_gap << endl;
		lout << "[SATrain] *** [End] ***" << endl;
	}
}

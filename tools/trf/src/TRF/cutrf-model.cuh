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

#ifndef _CUTRF_MODEL_CUH_
#define _CUTRF_MODEL_CUH_
#include "cutrf-feature.cuh"
#include "trf-model.h"

namespace cutrf
{
	typedef trf::PValue PValue;
	typedef trf::Prob Prob;
	typedef trf::LogP LogP;

#define FEAT_FIND_MAX 1000 // the max-size of the Find Array
#define FEAT_FIND_BUF(p) p + (blockIdx.x * blockDim.x + threadIdx.x) * FEAT_FIND_MAX

	/**
	* \class Model
	* \brief TRF model
	*/
	class Model
	{
	public:
		Feat m_feat;	 ///< hash all the features
		cu::Vec<PValue> m_value; ///< the value for each features

		int m_maxlen; ///< the maximum length of model, excluding <s> and </s>. The min-len = 1
		int m_maxOrder; ///< the maximum order of features
		cu::Vec<Prob> m_pi; ///< the prior length distribution \pi
		cu::Vec<LogP> m_logz; ///< the normalization constants log Z_l
		cu::Vec<LogP> m_zeta; ///< the estimated normalization constants \zeta_l (fix \zeta_1 = 0)

		Vocab m_vocab;

		cu::Mat<Prob> m_matLenJump; ///< [sample] used to propose a new length
		int m_maxSampleLen; ///< [sample] the maximum sample length, default = m_maxlen + 2

		int *m_pFindBuf; ///< the buffer for the Array in Find function
		cu::Mat<LogP> m_matClassSampleBuf; ///< the buffer used to sample class for each thread
		cu::Mat<LogP> m_matWordSampleBuf;  ///< the buffer used to sample word for each thread
		cu::Vec<RandState> m_vRandState;   ///< random state used to sample for each thread

	public:
		/// constructor
		__host__ __device__ Model() : m_maxlen(0), m_maxSampleLen(0) {};
		__host__ Model(trf::Model &m, int nMaxThread) { Copy(m, nMaxThread); }
		__host__ void Copy(trf::Model &m, int nMaxThread) {
			m_feat.Copy(*m.m_pFeat, nMaxThread);
			m_value.Copy(m.m_value);
			m_maxlen = m.m_maxlen;
			m_maxOrder = m.GetMaxOrder();
			m_pi.Copy(m.m_pi);
			m_logz.Copy(m.m_logz);
			m_zeta.Copy(m.m_zeta);
			m_vocab.Copy(*m.m_pVocab);
			m_matLenJump.Copy(m.m_matLenJump);
			m_maxSampleLen = m.m_maxSampleLen;

			CUDA_CALL(cudaMalloc(&m_pFindBuf, sizeof(int) * nMaxThread * FEAT_FIND_MAX));
			m_matClassSampleBuf.Reset(nMaxThread, m.m_pVocab->GetClassNum());
			m_matWordSampleBuf.Reset(nMaxThread, m_vocab.GetMaxWordInOneClass());

			m_vRandState.Reset(nMaxThread);
			Kernal_InitState<<<ceil(1.0*nMaxThread/128), 128>>>(m_vRandState.GetBuf(), nMaxThread, time(NULL));

		}
		__host__ void Release()
		{
			m_feat.Release();
			m_value.Reset();
			m_pi.Reset();
			m_logz.Reset();
			m_zeta.Reset();
			m_vocab.Release();
			m_matLenJump.Reset();
			
			CUDA_CALL(cudaFree(m_pFindBuf));
			m_matClassSampleBuf.Reset();
			m_matWordSampleBuf.Reset();
			m_vRandState.Reset();
		}

	public:
		__device__ LogP GetLogProb(Seq &seq, bool bNorm  = true )
		{
			cu::Array<int> afeat(FEAT_FIND_BUF(m_pFindBuf), FEAT_FIND_MAX);
			m_feat.Find(afeat, seq);

			LogP logSum = 0;
			for (int i = 0; i < afeat.GetNum(); i++) {
				logSum += m_value[afeat[i]];
			}

			if (bNorm) {
				int nLen = min(m_maxlen, seq.GetLen());
				logSum = logSum - m_logz[nLen] + Prob2LogP(m_pi[nLen]);
			}
			return logSum;
		}
		__device__ void Sample(Seq &seq)
		{
			LocalJump(seq);
			MarkovMove(seq);
		}
		__device__ void LocalJump(Seq &seq)
		{
			int nOldLen = seq.GetLen();
			int nNewLen = 0;
			LogP j1 = ProposeLength(nOldLen, nNewLen, true);
			LogP j2 = ProposeLength(nNewLen, nOldLen, false);

			if (nNewLen == nOldLen)
				return;

			LogP logpAcc = 0;
			if (nNewLen == nOldLen + 1) {
				LogP logpold = GetLogProb(seq);
				seq.Reset(nNewLen);
				LogP R = ProposeC0(seq.x[class_layer][nNewLen - 1], seq, nNewLen - 1, true);
				LogP G = SampleX(seq, nNewLen - 1);
				LogP logpnew = GetLogProb(seq);

				logpAcc = (j2 - j1) + logpnew - (logpold + R + G);
			}
			else if (nNewLen == nOldLen - 1) {
				LogP logpold = GetLogProb(seq);
				LogP R = ProposeC0(seq.x[class_layer][nOldLen - 1], seq, nOldLen - 1, false);
				LogP G = SampleX(seq, nOldLen - 1, false);

				seq.Reset(nNewLen);
				LogP logpnew = GetLogProb(seq);

				logpAcc = (j2 - j1) + logpnew + R + G - logpold;
			}
			else if (nNewLen != nOldLen){
				printf("[ERROR] [Model] Sample: nNewLen(%d) and nOldLen(%d)", nNewLen, nOldLen);
			}


			if (Acceptable(LogP2Prob(logpAcc), &m_vRandState[_TID])) {
				seq.Reset(nNewLen);
			}
			else {
				seq.Reset(nOldLen);
			}

		}
		/// [sample] Markov Move - perform the gibbs sampling
		__device__ void MarkovMove(Seq &seq)
		{
			/* Gibbs sampling */
			for (int nPos = 0; nPos < seq.GetLen(); nPos++) {
				SampleC(seq, nPos);
				SampleX(seq, nPos);
			}
		}
		/// [sample] Propose the length, using the variable m_matLenJump
		__device__ LogP ProposeLength(int nOld, int &nNew, bool bSample)
		{
			if (bSample) {
				nNew = LineSampling(m_matLenJump[nOld].GetBuf(), m_matLenJump[nOld].GetSize(), &m_vRandState[_TID]);
			}

			return Prob2LogP(m_matLenJump[nOld][nNew]);
		}
		/// [sample] Cuda application for trf::Model::GetReducedModelForC
		__device__ LogP GetReducedModelForC(Seq &seq, int nPos)
		{
			if (seq.x[class_layer][nPos] == VocabID_none)
				return 0;

			LogP logSum = 0;
			int nlen = seq.GetLen();
			int nMaxOrder = m_maxOrder;
			// class ngram features
			cu::Array<int> afeat(FEAT_FIND_BUF(m_pFindBuf), FEAT_FIND_MAX);
			for (int order = 1; order <= nMaxOrder; order++) {
				for (int i = max(0, nPos - order + 1); i <= min(nlen - order, nPos); i++) {
					m_feat.FindClass(afeat, seq, i, order);
				}
			}
			for (int i = 0; i < afeat.GetNum(); i++) {
				logSum += m_value[afeat[i]];
			}

			return logSum;
		}
		/// [sample] Cuda application for trf::Model::GetReducedModelForW
		__device__ LogP GetReducedModelForW(Seq &seq, int nPos)
		{
			LogP logSum = 0;
			int nlen = seq.GetLen();
			int nMaxOrder = m_maxOrder;
			// class ngram features
			cu::Array<int> afeat(FEAT_FIND_BUF(m_pFindBuf), FEAT_FIND_MAX);
			for (int order = 1; order <= nMaxOrder; order++) {
				for (int i = max(0, nPos - order + 1); i <= min(nlen - order, nPos); i++) {
					m_feat.FindWord(afeat, seq, i, order);
				}
			}
			for (int i = 0; i < afeat.GetNum(); i++) {
				logSum += m_value[afeat[i]];
			}

			return logSum;
		}
		/// [sample] A unnormalized reduced depending on nPos.
		__device__ LogP GetReducedModel(Seq &seq, int nPos)
		{
			LogP logSum = 0;
			int nlen = seq.GetLen();
			int nMaxOrder = m_maxOrder;
			// class ngram features
			cu::Array<int> afeat(FEAT_FIND_BUF(m_pFindBuf), FEAT_FIND_MAX);
			for (int order = 1; order <= nMaxOrder; order++) {
				for (int i = max(0, nPos - order + 1); i <= min(nlen - order, nPos); i++) {
					m_feat.Find(afeat, seq, i, order);
				}
			}
			for (int i = 0; i < afeat.GetNum(); i++) {
				logSum += m_value[afeat[i]];
			}

			return logSum;
		}
		/// [sample] given c_i, summate the probabilities of x_i, i.e. P(c_i)
		__device__ LogP GetMarginalProbOfC(Seq &seq, int nPos)
		{
			LogP resLogp = LOGP_ZERO;

			cu::VecShell<VocabID> vXs = m_vocab.GetWord(seq.x[class_layer][nPos]);

			VocabID saveX = seq.x[word_layer][nPos];
			for (int i = 0; i < vXs.GetSize(); i++) {
				seq.x[word_layer][nPos] = vXs[i];
				/* Only need to calculate the summation of weight depending on x[nPos], c[nPos] */
				/* used to sample the c_i */
				resLogp = Log_Sum(resLogp, GetReducedModel(seq, nPos));
				//resLogp = Log_Sum(resLogp, GetLogProb(seq, false));
			}
			seq.x[word_layer][nPos] = saveX;

			return resLogp;
		}
		/// [sample] Propose the c_{i} at position i. Then return the propose probability R(c_i|h_i,c_{other})
		__device__ LogP ProposeC0(VocabID &ci, Seq &seq, int nPos, bool bSample)
		{
			/* if there are no class, then return 0 */
			int nClassNum = m_vocab.GetClassNum();
			if (nClassNum == 0) {
				ci = VocabID_none;
				return 0;
			}

			cu::VecShell<LogP> vlogps = m_matClassSampleBuf[_TID];
			ProposeCProbs(vlogps, seq, nPos);

			if (bSample) {
				ci = LogLineSampling(vlogps.GetBuf(), vlogps.GetSize(), &m_vRandState[_TID]);
			}

			return vlogps[ci];
		}
		/// [sample] Return the propose distribution of c_i at position nPos
		__device__ void ProposeCProbs(cu::VecShell<LogP> logps, Seq &seq, int nPos)
		{
			VocabID savecid = seq.x[class_layer][nPos];
			int nClassNum = m_vocab.GetClassNum();
			for (int cid = 0; cid < nClassNum; cid++) {
				seq.x[class_layer][nPos] = cid;
				logps[cid] = GetReducedModelForC(seq, nPos);
			}
			seq.x[class_layer][nPos] = savecid;
			LogLineNormalize(logps.GetBuf(), nClassNum);
		}
		/// [sample] Sample the c_i at position nPos without x_i.
		__device__ void SampleC(Seq &seq, int nPos)
		{
			int nClassNum = m_vocab.GetClassNum();

			if (m_vocab.GetClassNum() == 0) {
				seq.x[class_layer][nPos] = VocabID_none;
				return;
			}

			/* Sample C0 */
			cu::VecShell<LogP> vlogps_c = m_matClassSampleBuf[_TID];
			ProposeCProbs(vlogps_c, seq, nPos);
			VocabID ci = seq.x[class_layer][nPos];
			VocabID C0 = LogLineSampling(vlogps_c.GetBuf(), vlogps_c.GetSize(), &m_vRandState[_TID]);
			LogP logpRi = vlogps_c[ci];
			LogP logpR0 = vlogps_c[C0];


			/* Calculate the probability p_t(h, c) */
			seq.x[class_layer][nPos] = ci;
			LogP Logp_ci = GetMarginalProbOfC(seq, nPos);
			seq.x[class_layer][nPos] = C0;
			LogP Logp_C0 = GetMarginalProbOfC(seq, nPos);

			LogP acclogp = logpRi + Logp_C0 - (logpR0 + Logp_ci);

			if (Acceptable(LogP2Prob(acclogp), &m_vRandState[_TID])) {
				seq.x[class_layer][nPos] = C0;
			}
			else {
				seq.x[class_layer][nPos] = ci;
			}
		}
		/// [sample] Sample the x_i at position nPos
		/* if bSample=ture, then sample x[nPos]. Otherwise only calculate the conditional probabilities of current x[nPos]. */
		__device__ LogP SampleX(Seq &seq, int nPos, bool bSample = true)
		{
			/*
			The function calculate G(x_i| x_{other}, h)
			if bSample is true, draw a sample for x_i;
			otherwise, only calcualte the conditional probability.
			*/
			cu::VecShell<VocabID> vXs = m_vocab.GetWord(seq.x[class_layer][nPos]);
			cu::Array<LogP> aLogps(m_matWordSampleBuf[_TID].GetBuf(), m_matWordSampleBuf.GetCol());

			VocabID nSaveX = seq.x[word_layer][nPos]; // save w[nPos]
			for (int i = 0; i < vXs.GetSize(); i++) {
				seq.x[word_layer][nPos] = vXs[i];
				/* To reduce the computational cost, instead of GetLogProb,
				we just need to calculate the summation of weight depending on x[nPos]
				*/
				aLogps[i] = GetReducedModelForW(seq, nPos);
			}
			LogLineNormalize(aLogps.GetBuf(), vXs.GetSize());

			int idx;
			if (bSample) {
				/* sample a value for x[nPos] */
				idx = LogLineSampling(aLogps.GetBuf(), vXs.GetSize(), &m_vRandState[_TID]);
				seq.x[word_layer][nPos] = vXs[idx];
			}
			else {
				// find nSave in the array
				idx = -1;
				for (int i = 0; i < vXs.GetSize(); i++) {
					if (vXs[i] == nSaveX) {
						idx = i;
						break;
					}
				}
				//idx = pXs->Find(nSaveX);
				seq.x[word_layer][nPos] = nSaveX;
				if (idx == -1) {
					printf("Can't find the VocabID(%d) in the the class(%d)\n", nSaveX, seq.x[class_layer][nPos]);
				}
			}

			return aLogps[idx];
		}
	};


	__global__ void Kernal_Sample(cu::Mat<int> matFind, cu::Array<cutrf::Seq> aSeq, cutrf::Model m)
	{
		int tid = _TID;

		if (tid < aSeq.GetNum()) {
			cu::Array<int> aFind(matFind[tid].GetBuf(), FEAT_FIND_MAX);

			m.Sample(aSeq[tid]);
			aFind.Add(aSeq[tid].GetLen()); ///< the first value is the length


			m.m_feat.Find(aFind, aSeq[tid]); /// the rest value is the finded features
			aFind.Add(VocabID_none);  ///< set as the end of the array.
		}
	}

	void cudaSample(cu::Mat<int> matFind, cu::Array<cutrf::Seq> dev_aSeq, cutrf::Model m, int nBlock, int nThreadPerBlock)
	{
		Kernal_Sample << <nBlock, nThreadPerBlock >> >(matFind, dev_aSeq, m);

// 		Kernal_LocalJump<<<nBlock, nThreadPerBlock>>>(dev_aSeq, m);
// 		
// 		for (int i = 0; i < m.m_maxSampleLen; i++) {
// 			Kernal_SampleC << <nBlock, nThreadPerBlock >> >(dev_aSeq, m, i);
// 		}
// 
// 		for (int i = 0; i < m.m_maxSampleLen; i++) {
// 			Kernal_SampleX << <nBlock, nThreadPerBlock >> >(dev_aSeq, m, i);
// 		}

		cudaError_t cudaStatus;
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return;
		}
	}

}

#endif
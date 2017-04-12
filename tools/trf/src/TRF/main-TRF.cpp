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
//#include "cutrf-model.cuh"
using namespace trf;

char *cfg_pathVocab = NULL;
char *cfg_pathModelRead = NULL;
char *cfg_pathModelWrite = NULL;

int cfg_nThread = 1;

char *cfg_pathTest = NULL;

/* lmscore */
char *cfg_pathNbest = NULL;
char *cfg_writeLmscore = NULL;
char *cfg_writeLmscoreDebug = NULL;
char *cfg_writeTestID = NULL;

/* normalization */
char *cfg_norm_method = NULL;
int cfg_nAIS_chain_num = 0;
int cfg_nAIS_inter_num = 0;
int cfg_norm_lenmin = 1;
int cfg_norm_lenmax = -1;

/* global normalization */
char *cfg_norm_global = NULL;

char *cfg_pathLenFile = NULL;

Option opt;
/* help */
const char *cfg_strHelp = "[Usage] : \n"
"Normalizing: \n"
"  trf -vocab [vocab] -read [model] -write [output model] -norm-method [Exact/AIS]\n"
"Calculate the global normalization constants: \n"
"  trf -vocab [vocab] -read [model] -write [output model] -norm-global [Exact/AIS] -norm-len-max [max_len]\n"
"Calculate log-likelihood:\n"
"  trf -vocab [vocab] -read [model] -test [txt-id-file]\n"
"language model rescoring:\n"
"  trf -vocab [vocab] -read [model] -nbest [nbest list] -lmscore [output lmscore]\n"
"Revise the length distribution pi:\n"
"  trf -vocab [vocab] -read [model] -write [output moddel] -len-file [a txt-id-file used to summary pi]\n"
;

#define  lout_exe lout<<"[TRF] "

double CalculateLL(Model &m, CorpusTxt *pCorpus, int nCorpusNum, double *pPPL = NULL);
void WordStr2ID(Array<VocabID> &aIDs, Array<String> &aStrs, LHash<const char*, VocabID> &vocabhash);
void LMRescore(Model &m, const char* pathTest);
void ModelNorm(Model &m, const char *type);
void ModelGlobal(Model &m, const char *type);
void ModelRevisePi(Model &m, const char *pathLenFile);

_wbMain
{ 
	opt.m_strOtherHelp = cfg_strHelp;
	opt.Add(wbOPT_STRING, "vocab", &cfg_pathVocab, "The vocabulary");
	opt.Add(wbOPT_STRING, "read", &cfg_pathModelRead, "Read the init model to train");
	opt.Add(wbOPT_STRING, "write", &cfg_pathModelWrite, "output the normalizaed model");
	opt.Add(wbOPT_INT, "thread", &cfg_nThread, "The thread number");

	opt.Add(wbOPT_STRING, "test", &cfg_pathTest, "test corpus (TXT)");

	opt.Add(wbOPT_STRING, "nbest", &cfg_pathNbest, "nbest list (kaldi output)");
	opt.Add(wbOPT_STRING, "lmscore", &cfg_writeLmscore, "[LMrescore] output the lmsocre");
	opt.Add(wbOPT_STRING, "lmscore-debug", &cfg_writeLmscoreDebug, "[LMrescore] output the lmscore of each word for word-level combination");
	opt.Add(wbOPT_STRING, "lmscore-test-id", &cfg_writeTestID, "[LMrescore] output the vocab-id of test file");

	opt.Add(wbOPT_STRING, "norm-method", &cfg_norm_method, "[Norm] method: Exact/AIS");
	opt.Add(wbOPT_INT, "AIS-chain", &cfg_nAIS_chain_num, "[AIS] the chain number");
	opt.Add(wbOPT_INT, "AIS-inter", &cfg_nAIS_inter_num, "[AIS] the intermediate distribution number");
	opt.Add(wbOPT_INT, "norm-len-min", &cfg_norm_lenmin, "[Norm] min-length");
	opt.Add(wbOPT_INT, "norm-len-max", &cfg_norm_lenmax, "[Norm] max-length");

	opt.Add(wbOPT_STRING, "norm-global", &cfg_norm_global, "[Global Norm] method: Exact/AIS");

	opt.Add(wbOPT_STRING, "len-file", &cfg_pathLenFile, "[Revise pi] a txt-id-file used to summary pi");

	opt.Parse(_argc, _argv);

	lout << "*********************************************" << endl;
	lout << "              TRF.exe                        " << endl;
	lout << "\t" << __DATE__ << "\t" << __TIME__ << "\t" << endl;
	lout << "**********************************************" << endl;

	omp_set_num_threads(cfg_nThread);
	lout << "[OMP] omp_thread = " << omp_get_max_threads() << endl;
	omp_rand(cfg_nThread);

	/// read model
	Vocab v(cfg_pathVocab);
	Model m(&v);
	lout_exe << "Read model: " << cfg_pathModelRead << endl;
	m.ReadT(cfg_pathModelRead);

	/* Operation 0: normalization */
	if (cfg_norm_method) {
		ModelNorm(m, cfg_norm_method);
	}

	/* Operation 0.5: normalization global */
	if (cfg_norm_global) {
		ModelGlobal(m, cfg_norm_global);
	}

	/* Operation 3: revise pi*/
	if (cfg_pathLenFile) {
		ModelRevisePi(m, cfg_pathLenFile);
	}

	/* Operation 1: calculate LL */
	if (cfg_pathTest) {
		CorpusTxt *p = new CorpusTxt(cfg_pathTest);
		double dPPL;
		double dLL = CalculateLL(m, p, p->GetNum(), &dPPL);
		lout_exe << "calculate LL of : " << cfg_pathTest << endl;
		lout_exe << "-LL = " << -dLL << endl;
		lout_exe << "PPL = " << dPPL << endl;
		SAFE_DELETE(p);
	}

	/* Operation 2: lmscore */
	if (cfg_pathNbest) {
		LMRescore(m, cfg_pathNbest);
	}
	

	/* write model */
	if (cfg_pathModelWrite) {
		lout_exe << "Write model: " << cfg_pathModelWrite << endl;
		m.WriteT(cfg_pathModelWrite);
	}

	return 1;
}

double CalculateLL(Model &m, CorpusTxt *pCorpus, int nCorpusNum, double *pPPL /*= NULL*/)
{
	Array<double> aLL(omp_get_max_threads());
	aLL.Fill(0);

	Array<int> aWords(omp_get_max_threads());
	aWords.Fill(0);
	Array<int> aSents(omp_get_max_threads());
	aSents.Fill(0);

	Array<VocabID> aSeq;
	lout.Progress(0, true, nCorpusNum - 1, "omp GetLL");
#pragma omp parallel for firstprivate(aSeq)
	for (int i = 0; i < nCorpusNum; i++) {
		pCorpus->GetSeq(i, aSeq);

		Seq seq;
		seq.Set(aSeq, m.m_pVocab);
		LogP logprob = m.GetLogProb(seq);

		aLL[omp_get_thread_num()] += logprob;
		aWords[omp_get_thread_num()] += aSeq.GetNum();
		aSents[omp_get_thread_num()] += 1;

#pragma omp critical
		lout.Progress();
	}

	double dLL = aLL.Sum() / nCorpusNum;
	int nSent = aSents.Sum();
	int nWord = aWords.Sum();
	lout_variable(nSent);
	lout_variable(nWord);
	if (pPPL) *pPPL = exp(-dLL * nSent / (nSent + nWord));
	return dLL;
}

void WordStr2ID(Array<VocabID> &aIDs, Array<String> &aStrs, LHash<const char*, VocabID> &vocabhash)
{
	for (int i = 0; i < aStrs.GetNum(); i++) {
		String wstr = aStrs[i];
		VocabID *pvid = vocabhash.Find(wstr.Toupper());
		if (pvid == NULL) { // cannot find the word, then find <UNK>
			// as word has been saved into hash with uppor style, 
			// then we need to find <UNK>, not <unk>.
			pvid = vocabhash.Find("<UNK>");
			if (!pvid) {
				lout_error("Can't find a vocab-id of " << wstr.GetBuffer());
			}
		}
		aIDs[i] = *pvid;
	}
}

void LMRescore(Model &m, const char* pathTest)
{
	Vocab *pV = m.m_pVocab;

	/// hash the vocab
	LHash<const char*, VocabID> vocabhash;
	bool bFound;
	for (int i = 0; i < pV->GetSize(); i++) {
		int *pVID = vocabhash.Insert(String(pV->GetWordStr(i)).Toupper(), bFound);
		if (bFound) {
			lout_exe << "Find words with same name but different id! (str="
				<< pV->GetWordStr(i) << " id=" << i << ")" << endl;
			exit(1);
		}
		*pVID = i;
	}

	/// rescore
	lout_exe << "Rescoring: " << pathTest << " ..." << endl;

	File fLmscore(cfg_writeLmscore, "wt");
	File fTestid(cfg_writeTestID, "wt");
	File file(pathTest, "rt");
	char *pLine;
	while (pLine = file.GetLine(true)) {
		String curLabel = strtok(pLine, " \t\n");
		String curSent = strtok(NULL, "\n");

		Array<String> aWordStrs;
		curSent.Split(aWordStrs, " \t\n");

		Array<VocabID> aWordIDs;
		WordStr2ID(aWordIDs, aWordStrs, vocabhash);

		Seq seq;
		seq.Set(aWordIDs, pV);
		LogP curLmscore = -m.GetLogProb(seq);

		/* output lmscore */
		fLmscore.Print("%s %lf\n", curLabel.GetBuffer(), curLmscore);
		/* output test-id */
		if (fTestid.Good()) {
			fTestid.Print("%s\t", curLabel.GetBuffer());
			fTestid.PrintArray("%d", aWordIDs.GetBuffer(), aWordIDs.GetNum());
		}
	}
}

void ModelNorm(Model &m, const char *type)
{
	String strType = type;
	strType.Tolower();
	if (strType == "exact") {
		lout_exe << "Exact Normalization..." << endl;
		m.ExactNormalize();
	}
	else if (strType == "ais") {
		lout_variable(m.ExactNormalize(1));
		if (cfg_nAIS_chain_num <= 0) {
			lout_exe << "[Input] AIS chain number = ";
			cin >> cfg_nAIS_chain_num;
		}
		if (cfg_nAIS_inter_num <= 0) {
			lout_exe << "[Input] AIS intermediate distribution number = ";
			cin >> cfg_nAIS_inter_num;
		}
		lout_exe << "AIS normalization..." << endl;
		lout_variable(cfg_nAIS_chain_num);
		lout_variable(cfg_nAIS_inter_num);

		srand(time(NULL));

		lout_variable(m.ExactNormalize(1));
		cfg_norm_lenmax = (cfg_norm_lenmax == -1) ? m.GetMaxLen() : cfg_norm_lenmax;
// 		for (int nLen = cfg_norm_lenmin; nLen <= cfg_norm_lenmax; nLen++)
// 		{
// 			lout_exe << "nLen = " << nLen << "/" << m.GetMaxLen() << ": ";
// 			m.AISNormalize(nLen, cfg_nAIS_chain_num, cfg_nAIS_inter_num);
// 			//cutrf::cudaModelAIS(m, nLen, cfg_nAIS_chain_num, cfg_nAIS_inter_num);
// 			lout << endl;
// 		}
		m.AISNormalize(cfg_norm_lenmin, cfg_norm_lenmax, cfg_nAIS_chain_num, cfg_nAIS_inter_num);
	}
	else {
		lout_error("Unknown method: " << type);
	}
}

void ModelGlobal(Model &m, const char *type)
{
	LogP logZ = LogP_zero;

	int old_len = m.GetMaxLen();
	if (cfg_norm_lenmax != -1) {
		lout_exe << "revise the maxlen = " << cfg_norm_lenmax << endl;
		// only revise the length-jump-distribution Gamma(k,j)
		m.ReviseLen(cfg_norm_lenmax);
	}

	String strType = type;
	strType.Tolower();
	if (strType == "exact") {
		lout_exe << "Exact Global Normalization..." << endl;
		Vec<LogP> zeta;
		zeta.Copy(m.m_zeta);
		m.ExactNormalize();

		logZ = LogP_zero;
		for (int j = 1; j <= m.GetMaxLen(); j++) {
			logZ = Log_Sum(logZ, Prob2LogP(m.m_pi[j]) - zeta[j] + m.m_logz[j] - m.m_logz[1]);
		}
	}
	else if (strType == "ais") {
		lout_variable(m.ExactNormalize(1));
		if (cfg_nAIS_chain_num <= 0) {
			lout_exe << "[Input] AIS chain number = ";
			cin >> cfg_nAIS_chain_num;
		}
		if (cfg_nAIS_inter_num <= 0) {
			lout_exe << "[Input] AIS intermediate distribution number = ";
			cin >> cfg_nAIS_inter_num;
		}
		lout_exe << "AIS normalization..." << endl;
		lout_variable(cfg_nAIS_chain_num);
		lout_variable(cfg_nAIS_inter_num);

		srand(time(NULL));
		logZ = m.AISNormGlobal(cfg_nAIS_chain_num, cfg_nAIS_inter_num);
	}
	else {
		lout_error("Unknown method: " << type);
	}

	if (cfg_norm_lenmax != -1) {
		m.ReviseLen(old_len);

		double pi_sum = 0;
		for (int j = cfg_norm_lenmax + 1; j <= old_len; j++) {
			pi_sum += m.m_pi[j];
		}
		lout_exe << "logZ before add pi_sum = " << logZ << endl;
		lout_exe << "add the exteral pi_sum = " << pi_sum << endl;
		logZ = Log_Sum(logZ, Prob2LogP(pi_sum));
	}

	lout_variable(logZ)
	for (int j = 1; j <= m.GetMaxLen(); j++) {
		m.m_logz[j] += logZ;
		m.m_zeta[j] += logZ;
	}

}

void ModelRevisePi(Model &m, const char *pathLenFile)
{
	lout << "Revise the length distribution pi..." << endl;
	int nMaxLen = m.GetMaxLen();
	Vec<Prob> vLen(nMaxLen+1);
	vLen.Fill(0);

	File file(pathLenFile, "rt");
	int nLine = 0;
	char *pLine;
	while (pLine = file.GetLine()) {
		nLine++;
		int nLen = 0;
		char *p = strtok(pLine, " \t\n");
		while (p) {
			nLen++;
			p = strtok(NULL, " \t\n");
		}
		nLen = min(nLen, nMaxLen);
		vLen[nLen] += 1;
	}
	vLen /= nLine;

	m.SetPi(vLen.GetBuf());
}

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


#ifdef _MLtrain

#include "trf-model.h"
#include "trf-ml-train.h"
#include <omp.h>
using namespace trf;

char *cfg_pathVocab = NULL;

int cfg_nFeatOrder = 2;
int cfg_nHNode = 2;
int cfg_nMaxLen = 10;

char *cfg_pathTrain = NULL;
char *cfg_pathValid = NULL;
char *cfg_pathTest = NULL;

char *cfg_pathModelRead = NULL;
char *cfg_pathModelWrite = "test.model";

int cfg_nIterTotalNum = 1000;
int cfg_nThread = 1;

Option opt;

_wbMain
{
	opt.Add(wbOPT_STRING, "vocab", &cfg_pathVocab, "The vocabulary");
	opt.Add(wbOPT_INT, "order", &cfg_nFeatOrder, "the ngram feature order (default=2)");
	opt.Add(wbOPT_STRING, "train", &cfg_pathTrain, "Training corpus (TXT)");
	opt.Add(wbOPT_STRING, "valid", &cfg_pathValid, "valid corpus (TXT)");
	opt.Add(wbOPT_STRING, "test", &cfg_pathTest, "test corpus (TXT)");

	opt.Add(wbOPT_STRING, "read", &cfg_pathModelRead, "Read the init model to train");
	opt.Add(wbOPT_STRING, "write", &cfg_pathModelWrite, "Output model");

	opt.Add(wbOPT_INT, "iter", &cfg_nIterTotalNum, "iter total number");
	opt.Add(wbOPT_INT, "thread", &cfg_nThread, "The thread number");

	opt.Parse();

	lout << "*********************************************" << endl;
	lout << "         TRF_train.exe { by Bin Wang }        " << endl;
	lout << "\t" << __DATE__ << "\t" << __TIME__ << "\t" << endl;
	lout << "**********************************************" << endl;

	omp_set_num_threads(cfg_nThread);
	lout << "[OMP] omp_thread = " << omp_get_max_threads() << endl;
	Title::SetGlobalTitle(String(cfg_pathModelWrite).FileName());

	Vocab *pv = new Vocab(cfg_pathVocab);
	Model m(pv, 10);
	if (cfg_pathModelRead) {
		m.ReadT(cfg_pathModelRead);
	}
	else {
		m.LoadFromCorpus(cfg_pathTrain, cfg_nFeatOrder);
	}
	m.WriteT(cfg_pathModelWrite);

	CorpusTxt *pTrain = (cfg_pathTrain) ? new CorpusTxt(cfg_pathTrain) : NULL;
	CorpusTxt *pValid = (cfg_pathValid) ? new CorpusTxt(cfg_pathValid) : NULL;
	CorpusTxt *pTest = (cfg_pathTest) ? new CorpusTxt(cfg_pathTest) : NULL;

	MLfunc func(&m, pTrain, pValid, pTest);
	func.m_pathOutputModel = cfg_pathModelWrite;


// 	Vec<double> vParams(func.GetParamNum());
// 	vParams.Fill(0.1);
// 
// 	Vec<double> vG1(func.GetParamNum());
// 	Vec<double> vG2(func.GetParamNum());
// 
// 	func.SetParam(vParams.GetBuf());
// 	func.GetGradient(vG1.GetBuf());
// 
// 	Vec<double> vP2(func.GetParamNum());
// 	double delta = 1e-3;
// 	for (int i = 2053; i < func.GetParamNum(); i++) {
// 		vP2.Copy(vParams);
// 		func.SetParam(vP2.GetBuf());
// 		double f1 = func.GetValue();
// 		vP2[i] += delta;
// 		func.SetParam(vP2.GetBuf());
// 		double f2 = func.GetValue();
// 		double g = (f2 - f1) / delta;
// 		lout << g << " " << vG1[i] << endl;
// 	}
// 	return 1;

	wb::LBFGS solve(&func);
	solve.m_nIterMax = cfg_nIterTotalNum; // fix the iteration number
	//solve.m_dGain = 1; // fixed the gain

	Vec<double> vInitParams(func.GetParamNum());
	vInitParams.Fill(0);

	if (cfg_pathModelRead) {
		func.GetParam(vInitParams.GetBuf());  ///< 使用当前的参数作为初始值
	}
	else {
		vInitParams.Fill(0);
	}


	solve.Run(vInitParams.GetBuf());

	// Finish
	func.SetParam(solve.m_pdRoot);
	m.WriteT(cfg_pathModelWrite);

	SAFE_DELETE(pTrain);
	SAFE_DELETE(pValid);
	SAFE_DELETE(pTest);

	SAFE_DELETE(pv);

	return 1;

	
	
}


#endif
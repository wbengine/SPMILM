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


#include "wb-option.h"

namespace wb
{
	Option::Option()
	{
		m_strOtherHelp = "";
		m_bOutputValues = true;
		m_bMustCommand = true;
	}
	Option::~Option()
	{
		for (int i = 0; i < m_allocedBufs.GetNum(); i++) {
			free(m_allocedBufs[i]);
		}
		m_allocedBufs.Clean();
	}
	void Option::Add(ValueType t, const char* pLabel, void *pAddress, const char* pDocMsg /* = NULL */)
	{
		m_opts.Add({ t, pLabel, pAddress, pDocMsg });
	}
	void Option::PrintUsage()
	{
		lout << "[Usage]:" << endl;
		lout << m_strOtherHelp.c_str() << endl;
		for (int i = 0; i < m_opts.GetNum(); i++)
		{
			lout << "-" << m_opts[i].pLabel << "\t" << m_opts[i].pDocMsg << " [default=";
			Print(m_opts[i].type, m_opts[i].pAddress);
			lout << "]" << endl;
		}
	}
	void Option::Print(ValueType type, void* pAddress)
	{
		switch (type)
		{
		case wbOPT_TRUE:
			lout << *((bool*)(pAddress));
			break;
		case wbOPT_FALSE:
			lout << *((bool*)(pAddress));
			break;
		case wbOPT_INT:
			lout << *((int*)(pAddress));
			break;
		case wbOPT_FLOAT:
			lout << *((float*)(pAddress));
			break;
		case wbOPT_STRING:
			lout << *((char**)(pAddress));
			break;
		}
	}
	void Option::PrintValue()
	{
		for (int i = 0; i < m_opts.GetNum(); i++)
		{
			lout << "-" << m_opts[i].pLabel << " =\t";
			Print(m_opts[i].type, m_opts[i].pAddress);
			lout << endl;
		}
	}
	void Option::Parse(const char *plabel, const char *pvalue)
	{
		int nOpt = -1;
		Opt_Struct *pOpt = NULL;
		for (int i = 0; i < m_opts.GetNum(); i++) {
			if (0 == strcmp(plabel + 1, m_opts[i].pLabel)) {
				nOpt = i;
				pOpt = &m_opts[i];
				break;
			}
		}

		if (!pOpt) { // unknown label
			if (strcmp(plabel, "-?") == 0) { // output the usage
				PrintUsage();
				exit(0);
			}
			else if (strcmp(plabel, "-log") == 0) { // revise the log file
				if (pvalue) {
					lout.ReFile(pvalue, true);
				}
				else {
					lout_error("no legal file name after -log");
				}
			}
			else {
				lout_error("Unknown label: " << plabel);
			}
		}
		else {
			if (pOpt->type != wbOPT_TRUE && pOpt->type != wbOPT_FALSE && !pvalue)
			{
				lout_error(plabel << " no corresponding value");
				exit(0);
			}

			switch (pOpt->type)
			{
			case wbOPT_TRUE:
				*((bool*)(pOpt->pAddress)) = true;
				break;
			case wbOPT_FALSE:
				*((bool*)(pOpt->pAddress)) = false;
				break;
			case wbOPT_INT:
				*((int*)(pOpt->pAddress)) = atoi(pvalue);
				break;
			case wbOPT_FLOAT:
				*((float*)(pOpt->pAddress)) = atof(pvalue);
				break;
			case wbOPT_STRING:
				/* for string, we allocate new memory */
				m_allocedBufs.Add() = strdup(pvalue);
				*((char**)(pOpt->pAddress)) = m_allocedBufs.End(); 
				break;
			}
		}
	}
	int Option::Parse(int argc /* = __argc */, char **argv /* = __argv */)
	{
		if (argc == 1 && m_bMustCommand) {
			PrintUsage();
			lout << "press any key to continue..." << endl;
			getch();
			exit(0);
		}

		register int CommandCur = 0;
		register char *CommandPtr = NULL;
		register int nOpt = 0;
		register Opt_Struct *pOpt = NULL;

		int nActiveOptNum = 0;
		for (CommandCur = 1; CommandCur < argc;)
		{
			char *pLabel = argv[CommandCur];
			char *pValue = NULL;
			if (CommandCur + 1 < argc) {
				pValue = argv[CommandCur + 1];
				if (*pValue == '-') {
					pValue = NULL;
				}
			}

			Parse(pLabel, pValue);

			nActiveOptNum++;
			if (pValue)
				CommandCur += 2;
			else
				CommandCur += 1;
		}

		if (m_bOutputValues)
			PrintValue();

		return nActiveOptNum;
	}
	int Option::Parse(const char *optfile)
	{
		ifstream file(optfile);
		
		int nActiveOptNum = 0;
		const int maxlength = 1024 * 4;
		char strline[maxlength];
		while (file.getline(strline, maxlength)) {
			char *pLabel = strtok(strline, "= \t");
			char *pContent = strtok(NULL, "= \t");
			Parse(pLabel, pContent);
			nActiveOptNum++;
		}
		if (nActiveOptNum == 0 && m_bMustCommand) {
			PrintUsage();
			lout << "press any key to continue..." << endl;
			getch();
			exit(0);
		}
		if (m_bOutputValues)
			PrintValue();

		return nActiveOptNum;
	}

}

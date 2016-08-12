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


#include "wb-win.h"
#ifdef __linux
#include "wb-linux.h"
#else
#include <windows.h>
#endif

namespace wb
{
	string Title::m_global_title = "";
	long long Title::m_precent_max = 0;
	long long Title::m_precent_cur = 0;
	int Title::m_precent_last = 0;
	string Title::m_precent_label = "";

	void Title::SetGlobalTitle(const char *pstr)
	{
		m_global_title = pstr;
	}
	const char* Title::GetGlobalTitle()
	{
		return m_global_title.c_str();
	}
	void Title::Puts(const char *pstr)
	{
		char command[10 + cn_title_max_len];
		sprintf(command, "title \"%s %s\"", m_global_title.c_str(), pstr);
		system(command);
	}
	void Title::Precent(long long n, bool bNew /* = false */, long long nTotal /* = 100 */, const char* label /* = "Process" */)
	{
		bool bUpdate = false;
		if (bNew)
		{
			m_precent_max = max(1LL, nTotal);
			m_precent_cur = n;
			m_precent_label = label;
			m_precent_last = 100 * m_precent_cur / m_precent_max;
			bUpdate = true;
		}
		else
		{
			m_precent_cur = n;
			int nNew = 100* m_precent_cur / m_precent_max;
			if (nNew > m_precent_last)
			{
				m_precent_last = nNew;
				bUpdate = true;
			}
		}

		if (bUpdate)
		{
			char outstr[cn_title_max_len];
			sprintf(outstr, "%s%3d%%", m_precent_label.c_str(), m_precent_last);
			Puts(outstr);
		}
	}
	void Title::Precent(ifstream &ifile, bool bNew /* = false */, const char* label /* = "" */)
	{
		if (bNew)
		{
			size_t nCur = ifile.tellg();
			ifile.seekg(0, ios_base::end);
			Precent(nCur, true, ifile.tellg(), label);
			ifile.seekg(nCur);
		}
		else
		{
			Precent(ifile.tellg());
		}
	}
	void Title::Precent(FILE *fp, bool bNew /* = false */, const char* label /* = "" */)
	{
		if (bNew)
		{
			long long nCur = _ftelli64(fp);
			_fseeki64(fp, 0, SEEK_END);
			Precent(nCur, true, _ftelli64(fp), label);
			_fseeki64(fp, nCur, SEEK_SET);
		}
		else
		{
			Precent(_ftelli64(fp));
		}
	}
	void Title::Fraction(long long n, bool bNew /* = false */, long long nTotal /* = 100 */, const char* label /* = "Process" */)
	{
		if (bNew)
		{
			m_precent_max = max(1LL, nTotal);
			m_precent_label = label;
		}
		m_precent_cur = n;

		char outstr[cn_title_max_len];
		sprintf(outstr, "%s%d/%d", m_precent_label.c_str(), n, m_precent_max);
		Puts(outstr);
	}

	Clock::Clock(void)
	{
		m_bWork = false;
		m_nBeginTime = 0;
		m_nEndTime = 0;
	}

	Clock::~Clock(void)
	{
		m_bWork = false;
		m_nBeginTime = 0;
		m_nEndTime = 0;
	}

	void Clock::Clean()
	{
		m_bWork = false;
		m_nBeginTime = 0;
		m_nEndTime = 0;
	}
	clock_t Clock::Begin()
	{
		m_bWork = true;
		return (m_nBeginTime = clock());
	}
	clock_t Clock::End()
	{
		if (m_bWork == false)
			return 0.0;
		m_bWork = false;
		m_nEndTime = clock();
		return m_nEndTime - m_nBeginTime;
	}
	clock_t Clock::Get()
	{
		return clock() - m_nBeginTime;
	}
	void Clock::Sleep(clock_t n)
	{
		Begin();
		while (Get() < n) {
		}
	}

#ifndef __linux
	Path::Path(const char *path/* =NULL */) : m_paFiles(NULL)
	{
		Reset(path);
	}
	Path::~Path()
	{
		Reset(NULL);
	}
	void Path::Reset(const char *path /* = NULL */)
	{
		// clean
		m_input = "";
		if (m_paFiles) {
			char *pp;
			while (m_paFiles->Out(&pp)) {
				delete [] pp;
			}
			m_paFiles->Clean();
		}

		// Reset
		if (path) {
			m_input = path;
			SearchFiles();
		}
	}
	bool Path::GetPath(char *path)
	{
		if (path == NULL)
			return false;
		if (m_paFiles == NULL)
		{
			SearchFiles();
		}

		char *pTemp;
		if (false == m_paFiles->Out(&pTemp))
			return false;
		strcpy(path, pTemp);
		delete [] pTemp;
		return true;
	}
	void Path::SearchFiles()
	{
		if (!m_paFiles)
			m_paFiles = new Queue<char*>;
		m_paFiles->Clean();

		// Find the paths
		Array<char*> aPaths;
		char *p = strtok(&m_input[0], "+\"");
		while (p) {
			aPaths.Add(p);
			p = strtok(NULL, "+\"");
		}

		// Find the files
		for (int i = 0; i < aPaths.GetNum(); i++)
		{
			//process fileName
			WIN32_FIND_DATAA fd;
			BOOL bRed = TRUE;
			HANDLE hFile = FindFirstFileA(aPaths[i], &fd);
			while (hFile != INVALID_HANDLE_VALUE && bRed)
			{
				char *p = new char[MAX_PATH_LEN];
				strcpy(p, aPaths[i]);
				char *temp = strrchr(p, '\\');
				if (temp == NULL)
					temp = p;
				else
					temp++;
				*temp = '\0';
				strcat(p, fd.cFileName);

				m_paFiles->In(p);

				bRed = FindNextFileA(hFile, &fd);
			}
		}
	}
#endif //__linux

	void Pause()
	{
		cout << "[Press any key to continue...]" << endl;
		getch();
	}
	void outPrecent(long long n, bool bNew /*= false*/, long long nTotal /*= 100*/, const char* title /*= "Process"*/)
	{
		static int snLastPrecent = 0;
		static long long snTotal = 100;
		static char strTitle[500];

		static clock_t begtime = 0;

		bool bUpdate = false;
		if (bNew)
		{
			if (nTotal > 0)
				snTotal = nTotal;
			snLastPrecent = n * 100 / snTotal;
			strcpy(strTitle, title);
			bUpdate = true;

			begtime = clock();
		}
		else
		{

			int nNew = (int)(1.0*n / snTotal * 100);
			if (nNew > snLastPrecent)
			{
				snLastPrecent = nNew;

				bUpdate = true;
			}
		}

		if (bUpdate)
		{
			if (bNew)
				cout << title << ":";
			else
				cout << "\b\b\b\b";

			cout.width(3);
			cout << snLastPrecent << "%";
		}
	}

	void outPrecent(ifstream &ifile, bool bNew, const char* title)
	{
		if (bNew)
		{
			ifile.seekg(0, ios_base::end);
			outPrecent(0, true, ifile.tellg(), title);
			ifile.seekg(0, ios_base::beg);
		}
		else
		{
			outPrecent(ifile.tellg());
		}
	}

	void outPrecent(FILE* fp, bool bNew, const char* title)
	{
		if (bNew)
		{
			long long iCur = _ftelli64(fp);
			_fseeki64(fp, 0, SEEK_END);
			outPrecent(iCur, true, _ftelli64(fp), title);
			_fseeki64(fp, iCur, SEEK_SET);
		}
		else
		{
			outPrecent(_ftelli64(fp));
		}
	}
	
}

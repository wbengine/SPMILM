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


#include "wb-log.h"

namespace wb
{
	Log lout;

	Log::Log()
	{
		m_bOutputCmd = true;
		m_bOutputLog = true;
		m_nLevel = 0;

#ifndef __linux
		char strFileName[255];
		strcpy(strFileName, strrchr(__argv[0], '\\') + 1);
		char *p = strstr(strFileName, ".");
		if (p)
			strcpy(p, ".log");
		else
			strcat(strFileName, ".log");

		m_fileLog.open(strFileName, ios_base::app);
		if (!m_fileLog.good())
		{
			cout << "Open Log file faled! [" << strFileName << "]" << endl;
			exit(0);
		}
#endif


		char strTime[128];
		char strDate[128];
		_strdate(strDate);
		_strtime(strTime);
		m_fileLog << endl;
		*this << "[Begin]\t" << strDate << " " << strTime << endl;

	}
	Log::~Log()
	{
		char strTime[128];
		char strDate[128];
		_strdate(strDate);
		_strtime(strTime);
		m_bOutputLog = true;
		m_bOutputCmd = true;
		*this << "[End]\t" << strDate << " " << strTime << endl;
	}

	/// ���¶�λlog������ļ���bNew�Ƿ����´����ļ���������Ѵ����ļ����ݣ�
	void Log::ReFile(const char *path, bool bNew/* = true*/)
	{
		m_bOutputCmd = true;
		m_bOutputLog = true;
		m_nLevel = 0;

		m_fileLog.close();
		if (bNew)
			m_fileLog.open(path);
		else
			m_fileLog.open(path, ios_base::app);

		if (!m_fileLog.good())
		{
			cout << "Open Log file faled! [" << path << "]" << endl;
			exit(0);
		}

		char strTime[128];
		char strDate[128];
		_strdate(strDate);
		_strtime(strTime);
		m_fileLog << endl;
		*this << "[ReFile]  bNew = " << bNew << endl;
		*this << "[Begin]\t" << strDate << " " << strTime << endl;
	}

	/// �Ƿ������cmd����
	bool &Log::bOutputCmd() { return m_bOutputCmd; }
	/// �Ƿ������log�ļ�
	bool &Log::bOutputLog() { return m_bOutputLog; }

	Log &Log::operator << (ostream& (*op) (ostream&)) { wbLog_Output(*op) }
	Log &Log::operator << (int x) { wbLog_Output(x) }
	Log &Log::operator << (short x) { wbLog_Output(x) }
	Log &Log::operator << (long x) { wbLog_Output(x) }
	Log &Log::operator << (long long x) { wbLog_Output(x) }
	Log &Log::operator << (unsigned int x) { wbLog_Output(x) }
	Log &Log::operator << (unsigned short x) { wbLog_Output(x) }
	Log &Log::operator << (unsigned long x) { wbLog_Output(x) }
	Log &Log::operator << (float x) { wbLog_Output(x) }
	Log &Log::operator << (double x) { wbLog_Output(x) }
	Log &Log::operator << (char x) { wbLog_Output(x) }
	Log &Log::operator << (const char* x) {
		//�ж�ָ���Ƿ�Ϸ�
		if (x) {
			wbLog_Output(x)
		}
		else {
			wbLog_Output("[NULL]")
		}
	}
	Log &Log::operator << (const void* x) {
		//�ж�ָ���Ƿ�Ϸ�
		if (x) {
			wbLog_Output(x)
		}
		else {
			wbLog_Output("[NULL]")
		}
	}
	Log &Log::operator << (bool x) {
		//��bool����ת�����ַ�����ӡ���
		if (x) {
			wbLog_Output("true")
		}
		else {
			wbLog_Output("false")
		}
	}
	Log &Log::operator << (string &x) {
		wbLog_Output(x.c_str());
	}
	/// ��������������һ��
	void Log::LevelDown() { m_nLevel++; }
	/// ��������������
	void Log::LevelUp() { m_nLevel = (m_nLevel <= 0) ? 0 : m_nLevel - 1; }

	void Log::Progress(long long n /* = -1 */, bool bInit /* = false */, long long total /* = 100 */, const char* head /* = "" */)
	{
		if (bInit) {
			m_bar.Reset(n, total, head, "[>]");
		}
		else {
			m_bar.Update(n);
		}
	}
	void Log::Progress(FILE *fp, bool bInit /* = false */, const char* head/* ="" */)
	{
		if (bInit)
		{
			long long nCur = _ftelli64(fp);
			_fseeki64(fp, 0, SEEK_END);
			Progress(nCur, true, _ftelli64(fp), head);
			_fseeki64(fp, nCur, SEEK_SET);
		}
		else
		{
			Progress(_ftelli64(fp));
		}
	}


	void ProgressBar::Reset(long long total /* = 100 */, const char *head /* = "" */, const char* sym /* = "[/* =]" */)
	{
		m_total = total;
		m_num = 0;
		memcpy(m_symbol, sym, sizeof(char) * 3);
		m_lastprec = -1;
		m_strhead = head;
		Update(0);
	}
	void ProgressBar::Reset(long long n, long long total /* = 100 */, const char *head /* = "" */, const char* sym /* = "[/* =]" */)
	{
		m_total = total;
		m_num = 0;
		memcpy(m_symbol, sym, sizeof(char) * 3);
		m_lastprec = -1;
		m_strhead = head;
		Update(n);
	}
	void ProgressBar::Update(long long n)
	{
		if (n < 0) {
			n = m_num;
			m_num++;
		}
		else {
			m_num = n;
		}
		int curprec = 100 * n / m_total;
		if (curprec > m_lastprec) {
			cout << "\r" << m_strhead.c_str() << " ";
			int barlen = (int)(1.0 * curprec / 100 * m_barmaxlen);
			cout << m_symbol[0];
			for (int i = 0; i < barlen; i++)
				cout << m_symbol[1];
			for (int i = 0; i < m_barmaxlen - barlen; i++)
				cout << " ";
			cout << m_symbol[2]<<" ";
			cout << setprecision(2) << setiosflags(ios::fixed) << 1.0 * n / m_total * 100 << "%" << setprecision(6);
			cout.flush();

			m_lastprec = curprec;
			if (curprec == 100) {
				cout << endl;
			}
		}
	}
}

# Mutual Funds Insight

�����M���̒l�����𕪐͂��A�`���[�g�\���E�T�}���[�����EAI����܂Œ񋟂��� Streamlit �A�v���P�[�V�����ł��BGoogle Spreadsheet �ɕۑ������e�����f�[�^���擾���A�e�N�j�J���w�W�̌v�Z������ԑ��֕��́ASlack �ʒm�AOpenAI ���g�������|�[�g�������s���܂��B

## ���݂̐i��
- Streamlit �x�[�X�� UI �������i�ڍו��́^�����ꗗ�^���֕��̓^�u�\���j
- Google Sheets �A�g���[�e�B���e�B�ƃe�N�j�J���w�W�v�Z���W���[���𐮔�
- OpenAI �𗘗p�����ڍו��͐����E�`���b�g�@�\��ǉ�
- Slack �ʒm�i�l�q�������������̕ύX���m�j�Ƒ��֕��͂̉���������
- Poetry / requirements.txt �o���ňˑ��֌W���Ǘ�

## �f�B���N�g���\��
```
mutual-funds-app/
������ data/                    # ���[�J�����ؗp�T���v���f�[�^
������ streamlit_app/
��   ������ app.py               # Streamlit �A�v���{��
��   ������ utils/               # Google Sheets / �w�W�v�Z / ���֕��� �ق�
������ src/app/                 # ����̍ė��p�������C�u�����R�[�h�iFastAPI �X�P���g���܂ށj
������ tests/                   # ���j�b�g�e�X�g
������ .env.example             # ���ϐ��e���v���[�g
������ requirements.txt
������ pyproject.toml
������ README.md
```

## �Z�b�g�A�b�v�菇
1. **Python 3.11 �ȏ�**���C���X�g�[�����܂��B
2. ���z�����쐬���ėL�������܂��B
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
3. �ˑ��p�b�P�[�W���C���X�g�[�����܂��B
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   # �܂��� Poetry �𗘗p����ꍇ�͎���2�s
   pip install poetry
   poetry install
   ```

## ���ϐ��̐ݒ�
`.env.example` �� `.env` �ɃR�s�[���A�ȉ���ݒ肵�Ă��������B
- `GOOGLE_SERVICE_ACCOUNT_KEY` : Google Sheets API �p�T�[�r�X�A�J�E���g JSON �𕶎��񉻂�������
- `GOOGLE_SPREADSHEET_ID` : ���͑Ώۂ̃X�v���b�h�V�[�g ID�i����l��ύX����ꍇ�j
- `OPENAI_API_KEY` : OpenAI API �L�[
- `SLACK_WEBHOOK_URL` : �ʒm���s���ꍇ�ɐݒ�i�C�Ӂj

Streamlit �� `st.secrets` ���g���ꍇ�́A`[gcp_service_account]` �Z�N�V�����ɓ��� JSON ���i�[����Γ��삵�܂��B

## �A�v���̋N��
```bash
streamlit run streamlit_app/app.py
```
�N����A�u���E�U�ɕ\������郁�j���[���������I�����ĕ��͂�i�߂܂��BSlack �ʒm�e�X�g��V�~�����[�V�����́u�����ꗗ�v�^�u���痘�p�ł��܂��B

## �e�X�g
```bash
pytest
```

## ����̊g���A�C�f�A
- Google Sheets ���獂�l�E���l�Ȃǂ̗���擾���� DMI ���x������
- FastAPI �x�[�X�� API �����J���A�ʃt�����g�G���h��������p�ł���悤����
- OpenAI �v�����v�g��o�̓e���v���[�g�̃J�X�^�}�C�Y
- �f�[�^�L���b�V����o�b�N�O���E���h�X�V�̍œK��
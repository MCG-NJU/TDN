mkdir  ~/.pip/
echo [global] > ~/.pip/pip.conf
echo index-url=https://mirrors.aliyun.com/pypi/simple/  >> ~/.pip/pip.conf

pip install -r requirements.txt
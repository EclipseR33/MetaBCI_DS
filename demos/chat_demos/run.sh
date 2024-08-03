cd ./Langchain-Chatchat/frontend
# you may need install cnpm and execute "sudo cnpm install" first
#sudo cnpm run dev
sudo cnpm run dev
cd ../../
cd ./Langchain-Chatchat/libs/chatchat-server/chatchat
python cli.py --api
cd ../../../../
python web.py

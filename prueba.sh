mkdir html2
rm -r html2/*
while read url
do
  FILE=./html2/`echo $url | sed -e "s/\//_/g"`.html
  echo "Descargando $url"
  wget -A htm,html,txt,shtml -a descargas2.log -O $FILE "$url"
done < descargas3.lnk
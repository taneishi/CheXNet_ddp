#!/bin/bash
# Download the 12 tgz files in batches

index=1
# URLs for the tar.gz files
for link in \
    https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz \
    https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz
do
    seq=$(printf "%02d" ${index})
    fn="images_${seq}.tar.gz"
    echo "wget -c -O ${fn} ${link}"
    wget -c -O ${fn} ${link}
    index=$((index+1))
done

echo "Download complete. Please check the checksums"
sha256sum -c SHA256_checksums.txt

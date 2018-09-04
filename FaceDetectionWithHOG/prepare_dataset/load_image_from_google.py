from bs4 import BeautifulSoup
import urllib3
import os
import json


def load(query):
    query='+'.join(query.split())
    url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"

    prefix = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 "
    suffix = "(KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
    header={'User-Agent': prefix + suffix}

    HTTP = urllib3.PoolManager()
    soup = BeautifulSoup(HTTP.request('GET', url, headers=header).data)

    images=[]
    for a in soup.find_all("div",{"class":"rg_meta"}):
        json_a = json.loads(a.text)
        link , Type =json_a["ou"], json_a["ity"]
        images.append((link,Type))

    print("there are total" , len(images),"images")

    path = "images_from_google"
    if not os.path.exists(path):
            os.mkdir(path)
    path = os.path.join(path, query.split('+')[0])

    if not os.path.exists(path):
            os.mkdir(path)
    cnt = 0
    for i, (img , Type) in enumerate(images):
        try:
            raw_img = HTTP.request('GET', img)
            img_name = query + "_" + str(cnt) + "." + (Type if len(Type) else "jpg")
            with open(os.path.join(path, img_name), "wb") as f:
                f.write(raw_img.data)
                cnt += 1
        except Exception as e:
            print("could not load : " + img)
            print(e)

if __name__ == "__main__":
    queries = ["city", "nature", "office"]
    for query in queries[2:]:
        load(query)
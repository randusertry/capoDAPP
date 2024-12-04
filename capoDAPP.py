import streamlit as st
import spacy
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
import lxml
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


url=st.secrets["url"]
html = requests.get(url).text
soup = BeautifulSoup(html, "lxml")
tables = soup.find_all("table")
real_rows=[]
for table in tables:
    rows=[[td.text.lower() for td in row.find_all("td")] for row in table.find_all("tr")]
    for r in rows:
        if len(r)>=3:
            real_row=r[:3]
            if real_row.count("")<3:
                if not real_row[0].startswith("\xa0"):
                    real_rows.append(real_row)

aller_df=pd.DataFrame(columns=real_rows[0])
for r in real_rows[1:]:
    aller_df.loc[len(aller_df)]=r
aller_df["nome"].fillna(method="ffill", inplace=True)

allergies=[a for a in aller_df["allergie"].unique() if a!=""]
avoidables=[a for a in aller_df["evitati"].unique()if a!=""]

st.write("Ciao, per la spesa :red[EVITA O SEGNALA TEMPESTIVAMENTE]:")
alls=""
for a in allergies:
    alls+=f"<li font-size=20px;>{a}</li>"
st.markdown(f"<ul>{alls}</ul>",unsafe_allow_html=True)

with st.sidebar:
    for name in aller_df["nome"].unique():
        alls_pers=[a for a in aller_df[aller_df["nome"]==name]["allergie"].unique() if a !=""]
        if alls_pers:
            with st.popover(name.upper()):
                st.write(f"{name} non può mangiare:")
                st.write(alls_pers)


st.write("Ricorda di segnalare i piatti che contengono:")

avoids=""
for a in avoidables:
    avoids+=f"<li font-size=20px;>{a}</li>"
st.markdown(f"<ul>{avoids}</ul>",unsafe_allow_html=True)


image_uploader=st.file_uploader("Sei in dubbio? Controlla se un prodotto è sicuro per tutti, carica una foto",accept_multiple_files=False)
url_uploader=st.text_input("Sei in dubbio? Controlla se una ricetta è sicura per tutti, carica il link")

if image_uploader:
    if st.button("Analizza la foto",key="imgbtn"):
        try:
            image = Image.open(image_uploader)
            processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
            model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
            pixel_values = processor(images=image, return_tensors="pt").pixel_values

            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            st.write(generated_text)
            try:
                nlp=spacy.load("it_core_news_sm")
                doc=nlp(generated_text)
                doc_lemmas=[t.lemma for t in doc]
                alle_lemmas=[t.lemma for t in nlp(" ".join(aller_df["allergie"].unique()))]+[t.lemma for t in nlp(" ".join(aller_df["evitati"].unique()))]
                alle_dict=[{t.lemma:t.text} for t in nlp(" ".join(aller_df["allergie"].unique()))]
                avoid_dict=[{t.lemma:t.text} for t in nlp(" ".join(aller_df["evitati"].unique()))]
                present_lemmas=[t for t in alle_lemmas if t in doc_lemmas]
                list_present_words=[[a[t] for a in alle_dict if t in list(a.keys())] for t in present_lemmas if t!=" "]+[[a[t] for a in avoid_dict if t in list(a.keys())] for t in present_lemmas if t!=" "]
                present_words=[]
                for el in list_present_words:
                    present_words+=el
                present_words=[el for el in present_words if el !=" "]            
                name_mask= (aller_df["allergie"].isin(present_words)) | (aller_df["evitati"].isin(present_words))
                names_toadv='\n'.join([n.upper() for n in aller_df[name_mask]["nome"].unique()])
                if present_lemmas:
                    st.write(":red[ATTENZIONE] potrebbe esserci:")
                    allergos=""
                    for t in present_words:
                        allergos+=f"<li color='red';>{t}</li>"
                    st.markdown(f"<ul>{allergos}</ul>",unsafe_allow_html=True)
                    st.write(f"Si prega di avvisare:\n {names_toadv}")

                    
            except:
                st.warning("Ah soccmel, qualcosa non va")
        except:
            st.warning("Ah soccmel, qualcosa non va")

if url_uploader:
    if st.button("Analizza la ricetta"):
        try:
            h=requests.get(url_uploader).text.lower()
            nlp=spacy.load("it_core_news_sm")
            doc=nlp(h)
            print(doc)
            doc_lemmas=[t.lemma for t in doc]
            alle_lemmas=[t.lemma for t in nlp(" ".join(aller_df["allergie"].unique()))]+[t.lemma for t in nlp(" ".join(aller_df["evitati"].unique()))]
            alle_dict=[{t.lemma:t.text} for t in nlp(" ".join(aller_df["allergie"].unique()))]
            avoid_dict=[{t.lemma:t.text} for t in nlp(" ".join(aller_df["evitati"].unique()))]
            present_lemmas=[t for t in alle_lemmas if t in doc_lemmas]
            list_present_words=[[a[t] for a in alle_dict if t in list(a.keys())] for t in present_lemmas if t!=" "]+[[a[t] for a in avoid_dict if t in list(a.keys())] for t in present_lemmas if t!=" "]
            present_words=[]
            for el in list_present_words:
                present_words+=el
            present_words=[el for el in present_words if el !=" "]
            name_mask= (aller_df["allergie"].isin(present_words)) | (aller_df["evitati"].isin(present_words))
            names_toadv='\n'.join([n.upper() for n in aller_df[name_mask]["nome"].unique()])
            if present_lemmas:
                st.write(":red[ATTENZIONE] potrebbe esserci:")
                allergos=""
                for t in present_words:
                    allergos+=f"<li color='red';>{t}</li>"
                st.markdown(f"<ul>{allergos}</ul>",unsafe_allow_html=True)
                st.write(f"Si prega di avvisare:\n {names_toadv}")
            else:
                st.write("Sembra che vada tutto :green[BENE], ricordati comunque di fare attenzione alla lista comprensiva più in alto nella pagina")

        except:
            st.warning("Ah soccmel, qualcosa non va")




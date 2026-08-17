# Nidaamka Saadaalinta Deeqda Waxbarasho (Machine Learning System) - Astaamaha Ugu Muhiimsan

Dukumeentigan wuxuu qeexayaa shaqooyinka asaasiga ah iyo astaamaha ugu muhiimsan ee Nidaamka Saadaalinta U-qalmida Deeqda Waxbarasho (Scholarship Eligibility Prediction System). Nidaamkan wuxuu isku xiraa XGBoost Machine Learning Model iyo web application, si uu u siiyo ardayda hab fudud oo ay deeqda ku dalbadaan, maamulkana u siiyo awood ay arjiyada ku maamulaan.

## 1. Doorka Isticmaalayaasha & Ilaalinta Xogta (Authentication)
* **Qaybta Ardayda:** Hab aamin ah oo ardayda ay isaga diiwaangelin karaan (Sign-up) una soo geli karaan (Login) nidaamka.
* **Qaybta Maamulka (Admin):** Bog gaar ah oo ammaan ah oo maamulka nidaamku ay ka soo galaan, kaas oo u baahan aqoonsi maamul oo gaar ah.

## 2. Astaamaha Asaasiga Ah Ee Application-ka
* **Xulashada Kulliyadda & Waaxda:** Nidaam u oggolaanaya ardayda inay doortaan kulliyadooda iyo qaybta ay rabaan inay dhigtaan.
* **Foomka Codsiga Deeqda Waxbarasho:** Foom dhammaystiran oo xogta ardayga ee aqoonta iyo midda shakhsiyeedba looga baahan yahay si saadaalinta loo sameeyo.
* **Machine Learning Prediction (Real-time Prediction):** Qiimayn degdeg ah oo lagu sameynayo in ardaygu u qalmo deeqda waxbarasho iyadoo la isticmaalayo XGBoost Model.
* **Faah-faahinta XGBoost Model (SHAP Explainability):** Bog gaar ah oo sharraxaya sababta XGBoost Model u sameeyey prediction-ka iyadoo la adeegsanayo nidaamka SHAP, taas oo daah-furnaan buuxda siinaysa natiijada model-ka.

## 3. Qaybta Maamulka (Admin Dashboard)
* **Maamulida Xogta Codsadayaasha:** Bog dhexe oo kulminaya dhammaan arjiyada la soo diray iyo Machine Learning Prediction-ka mid kasta.
* **Maamul Sare Oo Xogta Ah:** Awoodo dheeraad ah oo isugu jira raadin (Search), shaandhayn (Filter), xaqiijin gacan-ku-qabasho ah (Verify), iyo tirtiridda (Delete) xogta codsadayaasha.
* **Ka Hortagga Codsiyada Isku-noqnoqda:** Ardaygu hal codsi oo keliya ayuu u gudbin karaa kulliyad kasta inta lagu jiro scholarship cycle-ka furan.
* **Audit Log:** Nidaamku wuxuu kaydiyaa diiwaanka maamulka marka codsi la xaqiijiyo ama la tirtiro.
* **La Bixida Xogta (Data Export):** Awood u oggolaanaysa maamulka in ay xogta ardayda u badalaan qaabka CSV si warbixin ahaan loogu isticmaalo meelo kale.

## 4. Qalabka Iyo Tignoolajiyada La Isticmaalay (Tech Stack)
* **Backend:** Python oo lagu kabay qaab-dhismeedka (framework) Flask.
* **Xog-kaydiye (Database):** SQLite Database oo ah mid maxalli ah oo aamin ah, iyadoo la isticmaalayo Flask-SQLAlchemy.
* **Machine Learning:** Scikit-learn iyo XGBoost oo loo adeegsaday dhismaha model-ka saadaasha, si toos ahna loogu xiray web application-ka.
* **Frontend:** Nidaam waji qurux badan oo ku shaqaynaya aalad kasta (Responsive), laguna dhisay HTML, CSS, Bootstrap, iyo Chart.js oo loo isticmaalay in sawiro (charts) xogta lagu muujiyo.

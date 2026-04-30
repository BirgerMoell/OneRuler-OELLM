"""Language-specific references for mHELMET.

The entries are intentionally compact: they provide stable native-language
entities, places, and outcome words that can be recombined into deterministic
gold references for every task.  They are not a substitute for a curated corpus,
but they make the benchmark genuinely multilingual instead of English-only with
language-tagged noise.
"""

from __future__ import annotations


BASE_PROMPTS = {
    "recall": (
        "Read the text and remember the identifier facts.\n\n"
        "<text>\n{context}\n</text>\n\n"
        "<Question> What code is assigned to {query}? </Question>\n"
        "Answer with only the code."
    ),
    "rag": (
        "Use the retrieved passages to answer the question.\n\n"
        "{context}\n\n"
        "<Question> Which city hosted the {query} meeting? </Question>\n"
        "Answer with only the city name."
    ),
    "rerank": (
        "Rank the candidate passages by relevance to the question.\n\n"
        "<Question> Which passage identifies the archive key for {query}? </Question>\n\n"
        "{context}\n\n"
        "Return only the passage labels in best-to-worst order, separated by commas."
    ),
    "cite": (
        "Answer the question using the passages and cite the supporting passage label.\n\n"
        "{context}\n\n"
        "<Question> What is the registry value for {query}? </Question>\n"
        "Return the value followed by the citation label in brackets."
    ),
    "longqa": (
        "Read the long document and answer the question.\n\n"
        "<document>\n{context}\n</document>\n\n"
        "<Question> What verification phrase is linked to {query}? </Question>\n"
        "Answer with only the phrase."
    ),
    "summ": (
        "Summarize the document in exactly three bullet points. Preserve the key names, "
        "numbers, and outcomes.\n\n"
        "<document>\n{context}\n</document>"
    ),
    "icl": (
        "Infer the mapping from the examples, then answer the final item.\n\n"
        "{context}\n\n"
        "Final item: {query}\nAnswer with only the mapped label."
    ),
}


PROMPTS = {
    "bg": {
        "recall": "Прочетете текста и запомнете идентификационните данни.\n\n<text>\n{context}\n</text>\n\n<Question> Кой код е присвоен на {query}? </Question>\nОтговорете само с кода.",
        "rag": "Използвайте извлечените пасажи, за да отговорите на въпроса.\n\n{context}\n\n<Question> Кой град беше домакин на срещата за {query}? </Question>\nОтговорете само с името на града.",
        "rerank": "Подредете пасажите по релевантност към въпроса.\n\n<Question> Кой пасаж посочва архивния ключ за {query}? </Question>\n\n{context}\n\nВърнете само етикетите на пасажите от най-добър към най-слаб, разделени със запетаи.",
        "cite": "Отговорете на въпроса, като използвате пасажите, и цитирайте подкрепящия етикет.\n\n{context}\n\n<Question> Каква е регистровата стойност за {query}? </Question>\nВърнете стойността, последвана от етикета в скоби.",
        "longqa": "Прочетете дългия документ и отговорете на въпроса.\n\n<document>\n{context}\n</document>\n\n<Question> Коя проверочна фраза е свързана с {query}? </Question>\nОтговорете само с фразата.",
        "summ": "Обобщете документа в точно три точки. Запазете ключовите имена, числата и резултатите.\n\n<document>\n{context}\n</document>",
        "icl": "Изведете съответствието от примерите и отговорете на последния елемент.\n\n{context}\n\nПоследен елемент: {query}\nОтговорете само със съответния етикет.",
    },
    "cs": {
        "recall": "Přečtěte si text a zapamatujte si identifikační údaje.\n\n<text>\n{context}\n</text>\n\n<Question> Jaký kód je přiřazen k {query}? </Question>\nOdpovězte pouze kódem.",
        "rag": "Použijte vyhledané pasáže k zodpovězení otázky.\n\n{context}\n\n<Question> Které město hostilo setkání o {query}? </Question>\nOdpovězte pouze názvem města.",
        "rerank": "Seřaďte kandidátní pasáže podle relevance k otázce.\n\n<Question> Která pasáž identifikuje archivní klíč pro {query}? </Question>\n\n{context}\n\nVraťte pouze štítky pasáží od nejlepší po nejhorší, oddělené čárkami.",
        "cite": "Odpovězte na otázku pomocí pasáží a uveďte podpůrný štítek pasáže.\n\n{context}\n\n<Question> Jaká je registrová hodnota pro {query}? </Question>\nVraťte hodnotu následovanou citačním štítkem v hranatých závorkách.",
        "longqa": "Přečtěte si dlouhý dokument a odpovězte na otázku.\n\n<document>\n{context}\n</document>\n\n<Question> Jaká ověřovací fráze je spojena s {query}? </Question>\nOdpovězte pouze frází.",
        "summ": "Shrňte dokument přesně ve třech odrážkách. Zachovejte klíčové názvy, čísla a výsledky.\n\n<document>\n{context}\n</document>",
        "icl": "Odvoďte mapování z příkladů a poté odpovězte na poslední položku.\n\n{context}\n\nPoslední položka: {query}\nOdpovězte pouze přiřazeným štítkem.",
    },
    "da": {
        "recall": "Læs teksten og husk identifikationsoplysningerne.\n\n<text>\n{context}\n</text>\n\n<Question> Hvilken kode er tildelt {query}? </Question>\nSvar kun med koden.",
        "rag": "Brug de hentede passager til at besvare spørgsmålet.\n\n{context}\n\n<Question> Hvilken by var vært for mødet om {query}? </Question>\nSvar kun med bynavnet.",
        "rerank": "Rangér kandidatpassagerne efter relevans for spørgsmålet.\n\n<Question> Hvilken passage identificerer arkivnøglen for {query}? </Question>\n\n{context}\n\nReturnér kun passagelabels fra bedst til dårligst, adskilt med kommaer.",
        "cite": "Besvar spørgsmålet med passagerne og citér det understøttende passagelabel.\n\n{context}\n\n<Question> Hvad er registerværdien for {query}? </Question>\nReturnér værdien efterfulgt af citationslabel i klammer.",
        "longqa": "Læs det lange dokument og besvar spørgsmålet.\n\n<document>\n{context}\n</document>\n\n<Question> Hvilken verificeringsfrase er knyttet til {query}? </Question>\nSvar kun med frasen.",
        "summ": "Opsummér dokumentet i præcis tre punktopstillinger. Bevar nøglenavne, tal og resultater.\n\n<document>\n{context}\n</document>",
        "icl": "Udled mappingen fra eksemplerne, og besvar derefter det sidste element.\n\n{context}\n\nSidste element: {query}\nSvar kun med det tilknyttede label.",
    },
    "de": {
        "recall": "Lies den Text und merke dir die Kennungsdaten.\n\n<text>\n{context}\n</text>\n\n<Question> Welcher Code ist {query} zugewiesen? </Question>\nAntworte nur mit dem Code.",
        "rag": "Nutze die abgerufenen Passagen, um die Frage zu beantworten.\n\n{context}\n\n<Question> Welche Stadt war Gastgeberin des Treffens zu {query}? </Question>\nAntworte nur mit dem Stadtnamen.",
        "rerank": "Ordne die Kandidatenpassagen nach Relevanz für die Frage.\n\n<Question> Welche Passage identifiziert den Archivschlüssel für {query}? </Question>\n\n{context}\n\nGib nur die Passagenlabels von am besten bis am schlechtesten zurück, getrennt durch Kommas.",
        "cite": "Beantworte die Frage anhand der Passagen und zitiere das unterstützende Passagenlabel.\n\n{context}\n\n<Question> Welcher Registerwert gehört zu {query}? </Question>\nGib den Wert gefolgt vom Zitationslabel in Klammern zurück.",
        "longqa": "Lies das lange Dokument und beantworte die Frage.\n\n<document>\n{context}\n</document>\n\n<Question> Welche Verifizierungsphrase ist mit {query} verknüpft? </Question>\nAntworte nur mit der Phrase.",
        "summ": "Fasse das Dokument in genau drei Stichpunkten zusammen. Erhalte Schlüsselnamen, Zahlen und Ergebnisse.\n\n<document>\n{context}\n</document>",
        "icl": "Leite die Zuordnung aus den Beispielen ab und beantworte dann das letzte Element.\n\n{context}\n\nLetztes Element: {query}\nAntworte nur mit dem zugeordneten Label.",
    },
    "en": {
        "recall": "Read the text and remember the identifier facts.\n\n<text>\n{context}\n</text>\n\n<Question> What code is assigned to {query}? </Question>\nAnswer with only the code.",
        "rag": "Use the retrieved passages to answer the question.\n\n{context}\n\n<Question> Which city hosted the {query} meeting? </Question>\nAnswer with only the city name.",
        "rerank": "Rank the candidate passages by relevance to the question.\n\n<Question> Which passage identifies the archive key for {query}? </Question>\n\n{context}\n\nReturn only the passage labels in best-to-worst order, separated by commas.",
        "cite": "Answer the question using the passages and cite the supporting passage label.\n\n{context}\n\n<Question> What is the registry value for {query}? </Question>\nReturn the value followed by the citation label in brackets.",
        "longqa": "Read the long document and answer the question.\n\n<document>\n{context}\n</document>\n\n<Question> What verification phrase is linked to {query}? </Question>\nAnswer with only the phrase.",
        "summ": "Summarize the document in exactly three bullet points. Preserve the key names, numbers, and outcomes.\n\n<document>\n{context}\n</document>",
        "icl": "Infer the mapping from the examples, then answer the final item.\n\n{context}\n\nFinal item: {query}\nAnswer with only the mapped label.",
    },
    "es": {
        "recall": "Lee el texto y recuerda los datos de identificación.\n\n<text>\n{context}\n</text>\n\n<Question> ¿Qué código está asignado a {query}? </Question>\nResponde solo con el código.",
        "rag": "Usa los pasajes recuperados para responder la pregunta.\n\n{context}\n\n<Question> ¿Qué ciudad acogió la reunión sobre {query}? </Question>\nResponde solo con el nombre de la ciudad.",
        "rerank": "Ordena los pasajes candidatos por relevancia para la pregunta.\n\n<Question> ¿Qué pasaje identifica la clave de archivo de {query}? </Question>\n\n{context}\n\nDevuelve solo las etiquetas de mejor a peor, separadas por comas.",
        "cite": "Responde la pregunta usando los pasajes y cita la etiqueta de apoyo.\n\n{context}\n\n<Question> ¿Cuál es el valor de registro de {query}? </Question>\nDevuelve el valor seguido de la etiqueta de cita entre corchetes.",
        "longqa": "Lee el documento largo y responde la pregunta.\n\n<document>\n{context}\n</document>\n\n<Question> ¿Qué frase de verificación está vinculada a {query}? </Question>\nResponde solo con la frase.",
        "summ": "Resume el documento en exactamente tres viñetas. Conserva los nombres clave, números y resultados.\n\n<document>\n{context}\n</document>",
        "icl": "Deduce la asignación a partir de los ejemplos y responde el último elemento.\n\n{context}\n\nElemento final: {query}\nResponde solo con la etiqueta asignada.",
    },
    "fr": {
        "recall": "Lis le texte et mémorise les données d'identification.\n\n<text>\n{context}\n</text>\n\n<Question> Quel code est attribué à {query} ? </Question>\nRéponds uniquement avec le code.",
        "rag": "Utilise les passages récupérés pour répondre à la question.\n\n{context}\n\n<Question> Quelle ville a accueilli la réunion sur {query} ? </Question>\nRéponds uniquement avec le nom de la ville.",
        "rerank": "Classe les passages candidats selon leur pertinence pour la question.\n\n<Question> Quel passage identifie la clé d'archive de {query} ? </Question>\n\n{context}\n\nRenvoie uniquement les étiquettes, de la meilleure à la moins bonne, séparées par des virgules.",
        "cite": "Réponds à la question avec les passages et cite l'étiquette justificative.\n\n{context}\n\n<Question> Quelle est la valeur de registre de {query} ? </Question>\nRenvoie la valeur suivie de l'étiquette de citation entre crochets.",
        "longqa": "Lis le long document et réponds à la question.\n\n<document>\n{context}\n</document>\n\n<Question> Quelle phrase de vérification est liée à {query} ? </Question>\nRéponds uniquement avec la phrase.",
        "summ": "Résume le document en exactement trois puces. Conserve les noms clés, les nombres et les résultats.\n\n<document>\n{context}\n</document>",
        "icl": "Déduis la correspondance à partir des exemples, puis réponds au dernier élément.\n\n{context}\n\nDernier élément : {query}\nRéponds uniquement avec l'étiquette associée.",
    },
    "fi": {
        "recall": "Lue teksti ja paina tunnistetiedot mieleesi.\n\n<text>\n{context}\n</text>\n\n<Question> Mikä koodi on liitetty kohteeseen {query}? </Question>\nVastaa vain koodilla.",
        "rag": "Käytä haettuja katkelmia kysymykseen vastaamiseen.\n\n{context}\n\n<Question> Mikä kaupunki isännöi kokousta aiheesta {query}? </Question>\nVastaa vain kaupungin nimellä.",
        "rerank": "Järjestä ehdokaskatkelmat kysymyksen kannalta olennaisuuden mukaan.\n\n<Question> Mikä katkelma tunnistaa arkistoavaimen kohteelle {query}? </Question>\n\n{context}\n\nPalauta vain katkelmien tunnisteet parhaasta huonoimpaan pilkuilla erotettuina.",
        "cite": "Vastaa kysymykseen katkelmien avulla ja viittaa tukevaan katkelmatunnisteeseen.\n\n{context}\n\n<Question> Mikä on rekisteriarvo kohteelle {query}? </Question>\nPalauta arvo ja sen jälkeen viitetunniste hakasulkeissa.",
        "longqa": "Lue pitkä asiakirja ja vastaa kysymykseen.\n\n<document>\n{context}\n</document>\n\n<Question> Mikä varmennuslause liittyy kohteeseen {query}? </Question>\nVastaa vain lauseella.",
        "summ": "Tiivistä asiakirja täsmälleen kolmeen luetelmakohtaan. Säilytä avainnimet, numerot ja tulokset.\n\n<document>\n{context}\n</document>",
        "icl": "Päättele vastaavuus esimerkeistä ja vastaa sitten viimeiseen kohteeseen.\n\n{context}\n\nViimeinen kohde: {query}\nVastaa vain vastaavalla tunnisteella.",
    },
    "it": {
        "recall": "Leggi il testo e memorizza i dati identificativi.\n\n<text>\n{context}\n</text>\n\n<Question> Quale codice è assegnato a {query}? </Question>\nRispondi solo con il codice.",
        "rag": "Usa i passaggi recuperati per rispondere alla domanda.\n\n{context}\n\n<Question> Quale città ha ospitato l'incontro su {query}? </Question>\nRispondi solo con il nome della città.",
        "rerank": "Ordina i passaggi candidati per rilevanza rispetto alla domanda.\n\n<Question> Quale passaggio identifica la chiave d'archivio per {query}? </Question>\n\n{context}\n\nRestituisci solo le etichette dalla migliore alla peggiore, separate da virgole.",
        "cite": "Rispondi alla domanda usando i passaggi e cita l'etichetta di supporto.\n\n{context}\n\n<Question> Qual è il valore di registro per {query}? </Question>\nRestituisci il valore seguito dall'etichetta di citazione tra parentesi.",
        "longqa": "Leggi il documento lungo e rispondi alla domanda.\n\n<document>\n{context}\n</document>\n\n<Question> Quale frase di verifica è collegata a {query}? </Question>\nRispondi solo con la frase.",
        "summ": "Riassumi il documento in esattamente tre punti elenco. Mantieni nomi chiave, numeri e risultati.\n\n<document>\n{context}\n</document>",
        "icl": "Deduci la mappatura dagli esempi, poi rispondi all'elemento finale.\n\n{context}\n\nElemento finale: {query}\nRispondi solo con l'etichetta associata.",
    },
    "nl": {
        "recall": "Lees de tekst en onthoud de identificatiegegevens.\n\n<text>\n{context}\n</text>\n\n<Question> Welke code hoort bij {query}? </Question>\nAntwoord alleen met de code.",
        "rag": "Gebruik de opgehaalde passages om de vraag te beantwoorden.\n\n{context}\n\n<Question> Welke stad organiseerde de bijeenkomst over {query}? </Question>\nAntwoord alleen met de stadsnaam.",
        "rerank": "Rangschik de kandidaatpassages op relevantie voor de vraag.\n\n<Question> Welke passage identificeert de archiefsleutel voor {query}? </Question>\n\n{context}\n\nGeef alleen de passagelabels van best naar slechtst, gescheiden door komma's.",
        "cite": "Beantwoord de vraag met de passages en citeer het ondersteunende passagelabel.\n\n{context}\n\n<Question> Wat is de registerwaarde voor {query}? </Question>\nGeef de waarde gevolgd door het citatielabel tussen haken.",
        "longqa": "Lees het lange document en beantwoord de vraag.\n\n<document>\n{context}\n</document>\n\n<Question> Welke verificatiezin is gekoppeld aan {query}? </Question>\nAntwoord alleen met de zin.",
        "summ": "Vat het document samen in precies drie opsommingstekens. Behoud de sleutelnamen, nummers en uitkomsten.\n\n<document>\n{context}\n</document>",
        "icl": "Leid de koppeling af uit de voorbeelden en beantwoord daarna het laatste item.\n\n{context}\n\nLaatste item: {query}\nAntwoord alleen met het gekoppelde label.",
    },
    "no": {
        "recall": "Les teksten og husk identifikasjonsopplysningene.\n\n<text>\n{context}\n</text>\n\n<Question> Hvilken kode er tildelt {query}? </Question>\nSvar bare med koden.",
        "rag": "Bruk de hentede passasjene til å svare på spørsmålet.\n\n{context}\n\n<Question> Hvilken by var vertskap for møtet om {query}? </Question>\nSvar bare med bynavnet.",
        "rerank": "Ranger kandidatpassasjene etter relevans for spørsmålet.\n\n<Question> Hvilken passasje identifiserer arkivnøkkelen for {query}? </Question>\n\n{context}\n\nReturner bare passasjeetikettene fra best til dårligst, adskilt med komma.",
        "cite": "Svar på spørsmålet ved hjelp av passasjene og siter den støttende passasjeetiketten.\n\n{context}\n\n<Question> Hva er registerverdien for {query}? </Question>\nReturner verdien etterfulgt av siteringsetiketten i hakeparenteser.",
        "longqa": "Les det lange dokumentet og svar på spørsmålet.\n\n<document>\n{context}\n</document>\n\n<Question> Hvilken verifiseringsfrase er knyttet til {query}? </Question>\nSvar bare med frasen.",
        "summ": "Oppsummer dokumentet i nøyaktig tre punkter. Bevar nøkkelnavn, tall og resultater.\n\n<document>\n{context}\n</document>",
        "icl": "Utled koblingen fra eksemplene, og svar deretter på det siste elementet.\n\n{context}\n\nSiste element: {query}\nSvar bare med den koblede etiketten.",
    },
    "pl": {
        "recall": "Przeczytaj tekst i zapamiętaj dane identyfikacyjne.\n\n<text>\n{context}\n</text>\n\n<Question> Jaki kod przypisano do {query}? </Question>\nOdpowiedz wyłącznie kodem.",
        "rag": "Użyj pobranych fragmentów, aby odpowiedzieć na pytanie.\n\n{context}\n\n<Question> Które miasto było gospodarzem spotkania o {query}? </Question>\nOdpowiedz wyłącznie nazwą miasta.",
        "rerank": "Uszereguj fragmenty kandydackie według trafności względem pytania.\n\n<Question> Który fragment identyfikuje klucz archiwalny dla {query}? </Question>\n\n{context}\n\nZwróć tylko etykiety fragmentów od najlepszego do najgorszego, oddzielone przecinkami.",
        "cite": "Odpowiedz na pytanie na podstawie fragmentów i zacytuj wspierającą etykietę fragmentu.\n\n{context}\n\n<Question> Jaka jest wartość rejestrowa dla {query}? </Question>\nZwróć wartość, a po niej etykietę cytowania w nawiasach kwadratowych.",
        "longqa": "Przeczytaj długi dokument i odpowiedz na pytanie.\n\n<document>\n{context}\n</document>\n\n<Question> Jaka fraza weryfikacyjna jest powiązana z {query}? </Question>\nOdpowiedz wyłącznie frazą.",
        "summ": "Streść dokument dokładnie w trzech punktach. Zachowaj nazwy kluczowe, liczby i wyniki.\n\n<document>\n{context}\n</document>",
        "icl": "Wywnioskuj mapowanie z przykładów, a następnie odpowiedz na ostatni element.\n\n{context}\n\nOstatni element: {query}\nOdpowiedz wyłącznie przypisaną etykietą.",
    },
    "pt": {
        "recall": "Leia o texto e memorize os dados de identificação.\n\n<text>\n{context}\n</text>\n\n<Question> Que código está atribuído a {query}? </Question>\nResponda apenas com o código.",
        "rag": "Use as passagens recuperadas para responder à pergunta.\n\n{context}\n\n<Question> Que cidade acolheu a reunião sobre {query}? </Question>\nResponda apenas com o nome da cidade.",
        "rerank": "Ordene as passagens candidatas por relevância para a pergunta.\n\n<Question> Que passagem identifica a chave de arquivo de {query}? </Question>\n\n{context}\n\nDevolva apenas as etiquetas da melhor para a pior, separadas por vírgulas.",
        "cite": "Responda à pergunta usando as passagens e cite a etiqueta de apoio.\n\n{context}\n\n<Question> Qual é o valor de registo de {query}? </Question>\nDevolva o valor seguido da etiqueta de citação entre parênteses.",
        "longqa": "Leia o documento longo e responda à pergunta.\n\n<document>\n{context}\n</document>\n\n<Question> Que frase de verificação está ligada a {query}? </Question>\nResponda apenas com a frase.",
        "summ": "Resuma o documento em exatamente três tópicos. Preserve os nomes-chave, números e resultados.\n\n<document>\n{context}\n</document>",
        "icl": "Infira o mapeamento a partir dos exemplos e responda ao item final.\n\n{context}\n\nItem final: {query}\nResponda apenas com a etiqueta mapeada.",
    },
    "ro": {
        "recall": "Citește textul și reține datele de identificare.\n\n<text>\n{context}\n</text>\n\n<Question> Ce cod este atribuit pentru {query}? </Question>\nRăspunde doar cu codul.",
        "rag": "Folosește pasajele recuperate pentru a răspunde la întrebare.\n\n{context}\n\n<Question> Ce oraș a găzduit întâlnirea despre {query}? </Question>\nRăspunde doar cu numele orașului.",
        "rerank": "Ordonează pasajele candidate după relevanța față de întrebare.\n\n<Question> Ce pasaj identifică cheia de arhivă pentru {query}? </Question>\n\n{context}\n\nReturnează doar etichetele pasajelor de la cel mai bun la cel mai slab, separate prin virgule.",
        "cite": "Răspunde la întrebare folosind pasajele și citează eticheta pasajului justificativ.\n\n{context}\n\n<Question> Care este valoarea de registru pentru {query}? </Question>\nReturnează valoarea urmată de eticheta citării între paranteze drepte.",
        "longqa": "Citește documentul lung și răspunde la întrebare.\n\n<document>\n{context}\n</document>\n\n<Question> Ce frază de verificare este legată de {query}? </Question>\nRăspunde doar cu fraza.",
        "summ": "Rezumați documentul în exact trei puncte. Păstrați numele-cheie, numerele și rezultatele.\n\n<document>\n{context}\n</document>",
        "icl": "Deduce maparea din exemple, apoi răspunde la elementul final.\n\n{context}\n\nElement final: {query}\nRăspunde doar cu eticheta mapată.",
    },
    "ru": {
        "recall": "Прочитайте текст и запомните идентификационные данные.\n\n<text>\n{context}\n</text>\n\n<Question> Какой код присвоен {query}? </Question>\nОтветьте только кодом.",
        "rag": "Используйте найденные фрагменты, чтобы ответить на вопрос.\n\n{context}\n\n<Question> Какой город принимал встречу о {query}? </Question>\nОтветьте только названием города.",
        "rerank": "Упорядочьте фрагменты-кандидаты по релевантности вопросу.\n\n<Question> Какой фрагмент определяет архивный ключ для {query}? </Question>\n\n{context}\n\nВерните только метки фрагментов от лучшей к худшей, разделённые запятыми.",
        "cite": "Ответьте на вопрос, используя фрагменты, и процитируйте поддерживающую метку фрагмента.\n\n{context}\n\n<Question> Каково регистровое значение для {query}? </Question>\nВерните значение, затем метку цитирования в квадратных скобках.",
        "longqa": "Прочитайте длинный документ и ответьте на вопрос.\n\n<document>\n{context}\n</document>\n\n<Question> Какая проверочная фраза связана с {query}? </Question>\nОтветьте только фразой.",
        "summ": "Суммируйте документ ровно в трёх пунктах. Сохраните ключевые имена, числа и результаты.\n\n<document>\n{context}\n</document>",
        "icl": "Выведите соответствие из примеров, затем ответьте на последний элемент.\n\n{context}\n\nПоследний элемент: {query}\nОтветьте только соответствующей меткой.",
    },
    "sk": {
        "recall": "Prečítajte si text a zapamätajte si identifikačné údaje.\n\n<text>\n{context}\n</text>\n\n<Question> Aký kód je priradený k {query}? </Question>\nOdpovedzte iba kódom.",
        "rag": "Použite vyhľadané pasáže na zodpovedanie otázky.\n\n{context}\n\n<Question> Ktoré mesto hostilo stretnutie o {query}? </Question>\nOdpovedzte iba názvom mesta.",
        "rerank": "Zoraďte kandidátske pasáže podľa relevancie k otázke.\n\n<Question> Ktorá pasáž identifikuje archívny kľúč pre {query}? </Question>\n\n{context}\n\nVráťte iba štítky pasáží od najlepšej po najhoršiu, oddelené čiarkami.",
        "cite": "Odpovedzte na otázku pomocou pasáží a citujte podporný štítok pasáže.\n\n{context}\n\n<Question> Aká je registrová hodnota pre {query}? </Question>\nVráťte hodnotu nasledovanú citačným štítkom v hranatých zátvorkách.",
        "longqa": "Prečítajte si dlhý dokument a odpovedzte na otázku.\n\n<document>\n{context}\n</document>\n\n<Question> Aká overovacia fráza je spojená s {query}? </Question>\nOdpovedzte iba frázou.",
        "summ": "Zhrňte dokument presne v troch bodoch. Zachovajte kľúčové názvy, čísla a výsledky.\n\n<document>\n{context}\n</document>",
        "icl": "Odvoďte mapovanie z príkladov a potom odpovedzte na poslednú položku.\n\n{context}\n\nPosledná položka: {query}\nOdpovedzte iba priradeným štítkom.",
    },
    "sl": {
        "recall": "Preberite besedilo in si zapomnite identifikacijske podatke.\n\n<text>\n{context}\n</text>\n\n<Question> Katera koda je dodeljena {query}? </Question>\nOdgovorite samo s kodo.",
        "rag": "Uporabite pridobljene odlomke za odgovor na vprašanje.\n\n{context}\n\n<Question> Katero mesto je gostilo srečanje o {query}? </Question>\nOdgovorite samo z imenom mesta.",
        "rerank": "Razvrstite kandidatne odlomke po relevantnosti za vprašanje.\n\n<Question> Kateri odlomek identificira arhivski ključ za {query}? </Question>\n\n{context}\n\nVrnite samo oznake odlomkov od najboljšega do najslabšega, ločene z vejicami.",
        "cite": "Odgovorite na vprašanje z uporabo odlomkov in navedite podporno oznako odlomka.\n\n{context}\n\n<Question> Kakšna je registrska vrednost za {query}? </Question>\nVrnite vrednost, nato oznako citata v oglatih oklepajih.",
        "longqa": "Preberite dolg dokument in odgovorite na vprašanje.\n\n<document>\n{context}\n</document>\n\n<Question> Katera preveritvena fraza je povezana z {query}? </Question>\nOdgovorite samo s frazo.",
        "summ": "Povzemite dokument v natanko treh alinejah. Ohranite ključna imena, številke in rezultate.\n\n<document>\n{context}\n</document>",
        "icl": "Iz primerov izpeljite preslikavo in nato odgovorite na zadnji element.\n\n{context}\n\nZadnji element: {query}\nOdgovorite samo z dodeljeno oznako.",
    },
    "sv": {
        "recall": "Läs texten och kom ihåg identifieringsuppgifterna.\n\n<text>\n{context}\n</text>\n\n<Question> Vilken kod är tilldelad {query}? </Question>\nSvara endast med koden.",
        "rag": "Använd de hämtade passagerna för att besvara frågan.\n\n{context}\n\n<Question> Vilken stad var värd för mötet om {query}? </Question>\nSvara endast med stadsnamnet.",
        "rerank": "Rangordna kandidatpassagerna efter relevans för frågan.\n\n<Question> Vilken passage identifierar arkivnyckeln för {query}? </Question>\n\n{context}\n\nReturnera endast passageetiketterna från bäst till sämst, separerade med kommatecken.",
        "cite": "Besvara frågan med hjälp av passagerna och citera den stödjande passageetiketten.\n\n{context}\n\n<Question> Vilket registervärde gäller för {query}? </Question>\nReturnera värdet följt av citationsetiketten inom hakparenteser.",
        "longqa": "Läs det långa dokumentet och besvara frågan.\n\n<document>\n{context}\n</document>\n\n<Question> Vilken verifieringsfras är kopplad till {query}? </Question>\nSvara endast med frasen.",
        "summ": "Sammanfatta dokumentet i exakt tre punkter. Bevara nyckelnamn, siffror och resultat.\n\n<document>\n{context}\n</document>",
        "icl": "Härled mappningen från exemplen och besvara sedan det sista objektet.\n\n{context}\n\nSista objekt: {query}\nSvara endast med den mappade etiketten.",
    },
    "uk": {
        "recall": "Прочитайте текст і запам'ятайте ідентифікаційні дані.\n\n<text>\n{context}\n</text>\n\n<Question> Який код присвоєно {query}? </Question>\nВідповідайте лише кодом.",
        "rag": "Використайте знайдені уривки, щоб відповісти на запитання.\n\n{context}\n\n<Question> Яке місто приймало зустріч про {query}? </Question>\nВідповідайте лише назвою міста.",
        "rerank": "Впорядкуйте уривки-кандидати за релевантністю до запитання.\n\n<Question> Який уривок визначає архівний ключ для {query}? </Question>\n\n{context}\n\nПоверніть лише мітки уривків від найкращої до найгіршої, розділені комами.",
        "cite": "Дайте відповідь на запитання, використовуючи уривки, і процитуйте підтримувальну мітку уривка.\n\n{context}\n\n<Question> Яке реєстрове значення для {query}? </Question>\nПоверніть значення, а потім мітку цитування в квадратних дужках.",
        "longqa": "Прочитайте довгий документ і дайте відповідь на запитання.\n\n<document>\n{context}\n</document>\n\n<Question> Яка перевірочна фраза пов'язана з {query}? </Question>\nВідповідайте лише фразою.",
        "summ": "Підсумуйте документ рівно трьома пунктами. Збережіть ключові назви, числа та результати.\n\n<document>\n{context}\n</document>",
        "icl": "Виведіть відповідність із прикладів, а потім дайте відповідь для останнього елемента.\n\n{context}\n\nОстанній елемент: {query}\nВідповідайте лише відповідною міткою.",
    },
    "bs": {
        "recall": "Pročitajte tekst i zapamtite identifikacijske podatke.\n\n<text>\n{context}\n</text>\n\n<Question> Koji kod je dodijeljen za {query}? </Question>\nOdgovorite samo kodom.",
        "rag": "Koristite pronađene odlomke da odgovorite na pitanje.\n\n{context}\n\n<Question> Koji grad je bio domaćin sastanka o {query}? </Question>\nOdgovorite samo imenom grada.",
        "rerank": "Poredajte odlomke kandidate po relevantnosti za pitanje.\n\n<Question> Koji odlomak identifikuje arhivski ključ za {query}? </Question>\n\n{context}\n\nVratite samo oznake odlomaka od najbolje do najlošije, odvojene zarezima.",
        "cite": "Odgovorite na pitanje koristeći odlomke i citirajte podržavajuću oznaku odlomka.\n\n{context}\n\n<Question> Koja je registarska vrijednost za {query}? </Question>\nVratite vrijednost praćenu oznakom citata u uglastim zagradama.",
        "longqa": "Pročitajte dugi dokument i odgovorite na pitanje.\n\n<document>\n{context}\n</document>\n\n<Question> Koja verifikacijska fraza je povezana sa {query}? </Question>\nOdgovorite samo frazom.",
        "summ": "Sažmite dokument u tačno tri stavke. Sačuvajte ključna imena, brojeve i ishode.\n\n<document>\n{context}\n</document>",
        "icl": "Izvedite mapiranje iz primjera, zatim odgovorite na posljednju stavku.\n\n{context}\n\nPosljednja stavka: {query}\nOdgovorite samo mapiranom oznakom.",
    },
    "ca": {
        "recall": "Llegeix el text i recorda les dades d'identificació.\n\n<text>\n{context}\n</text>\n\n<Question> Quin codi està assignat a {query}? </Question>\nRespon només amb el codi.",
        "rag": "Fes servir els passatges recuperats per respondre la pregunta.\n\n{context}\n\n<Question> Quina ciutat va acollir la reunió sobre {query}? </Question>\nRespon només amb el nom de la ciutat.",
        "rerank": "Ordena els passatges candidats per rellevància respecte a la pregunta.\n\n<Question> Quin passatge identifica la clau d'arxiu de {query}? </Question>\n\n{context}\n\nRetorna només les etiquetes dels passatges de millor a pitjor, separades per comes.",
        "cite": "Respon la pregunta amb els passatges i cita l'etiqueta de suport.\n\n{context}\n\n<Question> Quin és el valor de registre de {query}? </Question>\nRetorna el valor seguit de l'etiqueta de citació entre claudàtors.",
        "longqa": "Llegeix el document llarg i respon la pregunta.\n\n<document>\n{context}\n</document>\n\n<Question> Quina frase de verificació està vinculada a {query}? </Question>\nRespon només amb la frase.",
        "summ": "Resumeix el document en exactament tres punts. Conserva els noms clau, els números i els resultats.\n\n<document>\n{context}\n</document>",
        "icl": "Dedueix el mapatge a partir dels exemples i respon l'element final.\n\n{context}\n\nElement final: {query}\nRespon només amb l'etiqueta mapada.",
    },
    "cy": {
        "recall": "Darllenwch y testun a chofiwch y data adnabod.\n\n<text>\n{context}\n</text>\n\n<Question> Pa god sydd wedi'i neilltuo i {query}? </Question>\nAtebwch gyda'r cod yn unig.",
        "rag": "Defnyddiwch y darnau a adferwyd i ateb y cwestiwn.\n\n{context}\n\n<Question> Pa ddinas a gynhaliodd y cyfarfod am {query}? </Question>\nAtebwch gydag enw'r ddinas yn unig.",
        "rerank": "Rhowch y darnau ymgeisiol mewn trefn yn ôl perthnasedd i'r cwestiwn.\n\n<Question> Pa ddarn sy'n adnabod yr allwedd archif ar gyfer {query}? </Question>\n\n{context}\n\nDychwelwch labeli'r darnau yn unig, o'r gorau i'r gwaethaf, wedi'u gwahanu gan atalnodau.",
        "cite": "Atebwch y cwestiwn gan ddefnyddio'r darnau a dyfynnwch label y darn ategol.\n\n{context}\n\n<Question> Beth yw gwerth y gofrestr ar gyfer {query}? </Question>\nDychwelwch y gwerth wedi'i ddilyn gan y label dyfynnu mewn cromfachau sgwâr.",
        "longqa": "Darllenwch y ddogfen hir ac atebwch y cwestiwn.\n\n<document>\n{context}\n</document>\n\n<Question> Pa ymadrodd dilysu sy'n gysylltiedig â {query}? </Question>\nAtebwch gyda'r ymadrodd yn unig.",
        "summ": "Crynhowch y ddogfen mewn union dri phwynt. Cadwch yr enwau allweddol, y rhifau a'r canlyniadau.\n\n<document>\n{context}\n</document>",
        "icl": "Casglwch y mapio o'r enghreifftiau, yna atebwch yr eitem olaf.\n\n{context}\n\nEitem olaf: {query}\nAtebwch gyda'r label wedi'i fapio yn unig.",
    },
    "el": {
        "recall": "Διαβάστε το κείμενο και θυμηθείτε τα στοιχεία ταυτοποίησης.\n\n<text>\n{context}\n</text>\n\n<Question> Ποιος κωδικός έχει ανατεθεί στο {query}; </Question>\nΑπαντήστε μόνο με τον κωδικό.",
        "rag": "Χρησιμοποιήστε τα ανακτημένα αποσπάσματα για να απαντήσετε στην ερώτηση.\n\n{context}\n\n<Question> Ποια πόλη φιλοξένησε τη συνάντηση για το {query}; </Question>\nΑπαντήστε μόνο με το όνομα της πόλης.",
        "rerank": "Ταξινομήστε τα υποψήφια αποσπάσματα κατά συνάφεια με την ερώτηση.\n\n<Question> Ποιο απόσπασμα προσδιορίζει το αρχειακό κλειδί για το {query}; </Question>\n\n{context}\n\nΕπιστρέψτε μόνο τις ετικέτες των αποσπασμάτων από την καλύτερη στη χειρότερη, χωρισμένες με κόμματα.",
        "cite": "Απαντήστε στην ερώτηση χρησιμοποιώντας τα αποσπάσματα και παραθέστε την υποστηρικτική ετικέτα αποσπάσματος.\n\n{context}\n\n<Question> Ποια είναι η τιμή μητρώου για το {query}; </Question>\nΕπιστρέψτε την τιμή ακολουθούμενη από την ετικέτα παραπομπής σε αγκύλες.",
        "longqa": "Διαβάστε το μεγάλο έγγραφο και απαντήστε στην ερώτηση.\n\n<document>\n{context}\n</document>\n\n<Question> Ποια φράση επαλήθευσης συνδέεται με το {query}; </Question>\nΑπαντήστε μόνο με τη φράση.",
        "summ": "Συνοψίστε το έγγραφο σε ακριβώς τρία σημεία. Διατηρήστε τα βασικά ονόματα, τους αριθμούς και τα αποτελέσματα.\n\n<document>\n{context}\n</document>",
        "icl": "Συμπεράνετε την αντιστοίχιση από τα παραδείγματα και μετά απαντήστε στο τελευταίο στοιχείο.\n\n{context}\n\nΤελευταίο στοιχείο: {query}\nΑπαντήστε μόνο με την αντιστοιχισμένη ετικέτα.",
    },
    "et": {
        "recall": "Lugege teksti ja jätke identifitseerimisandmed meelde.\n\n<text>\n{context}\n</text>\n\n<Question> Milline kood on määratud kirjele {query}? </Question>\nVastake ainult koodiga.",
        "rag": "Kasutage küsimusele vastamiseks leitud lõike.\n\n{context}\n\n<Question> Milline linn võõrustas kohtumist teemal {query}? </Question>\nVastake ainult linna nimega.",
        "rerank": "Järjestage kandidaat-lõigud küsimuse suhtes asjakohasuse järgi.\n\n<Question> Milline lõik tuvastab arhiivivõtme kirjele {query}? </Question>\n\n{context}\n\nTagastage ainult lõikude sildid parimast halvimani, eraldatuna komadega.",
        "cite": "Vastake küsimusele lõikude abil ja tsiteerige toetava lõigu silti.\n\n{context}\n\n<Question> Mis on registriväärtus kirjele {query}? </Question>\nTagastage väärtus, millele järgneb tsiteerimissilt nurksulgudes.",
        "longqa": "Lugege pikka dokumenti ja vastake küsimusele.\n\n<document>\n{context}\n</document>\n\n<Question> Milline kontrollifraas on seotud kirjega {query}? </Question>\nVastake ainult fraasiga.",
        "summ": "Võtke dokument kokku täpselt kolme punktina. Säilitage võtmenimed, numbrid ja tulemused.\n\n<document>\n{context}\n</document>",
        "icl": "Tuletage näidetest vastendus ja vastake seejärel viimasele üksusele.\n\n{context}\n\nViimane üksus: {query}\nVastake ainult vastendatud sildiga.",
    },
    "eu": {
        "recall": "Irakurri testua eta gogoratu identifikazio datuak.\n\n<text>\n{context}\n</text>\n\n<Question> Zein kode esleitu zaio {query} elementuari? </Question>\nErantzun kodearekin bakarrik.",
        "rag": "Erabili berreskuratutako pasarteak galderari erantzuteko.\n\n{context}\n\n<Question> Zein hirik hartu zuen {query} gaiari buruzko bilera? </Question>\nErantzun hiriaren izenarekin bakarrik.",
        "rerank": "Ordenatu hautagai-pasarteak galderarekiko garrantziaren arabera.\n\n<Question> Zein pasartek identifikatzen du {query} elementuaren artxibo-gakoa? </Question>\n\n{context}\n\nItzuli pasarte-etiketak bakarrik, onenetik txarrenera, komaz bereizita.",
        "cite": "Erantzun galderari pasarteak erabiliz eta aipatu euskarri-pasartearen etiketa.\n\n{context}\n\n<Question> Zein da {query} elementuaren erregistro-balioa? </Question>\nItzuli balioa eta ondoren aipu-etiketa kortxete artean.",
        "longqa": "Irakurri dokumentu luzea eta erantzun galderari.\n\n<document>\n{context}\n</document>\n\n<Question> Zein egiaztapen-esaldi dago lotuta {query} elementuarekin? </Question>\nErantzun esaldiarekin bakarrik.",
        "summ": "Laburtu dokumentua zehazki hiru puntutan. Mantendu gako-izenak, zenbakiak eta emaitzak.\n\n<document>\n{context}\n</document>",
        "icl": "Ondorioztatu mapaketa adibideetatik, eta erantzun azken elementuari.\n\n{context}\n\nAzken elementua: {query}\nErantzun mapatutako etiketarekin bakarrik.",
    },
    "ga": {
        "recall": "Léigh an téacs agus cuimhnigh ar na sonraí aitheantais.\n\n<text>\n{context}\n</text>\n\n<Question> Cén cód atá sannta do {query}? </Question>\nFreagair leis an gcód amháin.",
        "rag": "Úsáid na sleachta aisghafa chun an cheist a fhreagairt.\n\n{context}\n\n<Question> Cén chathair a d'óstáil an cruinniú faoi {query}? </Question>\nFreagair le hainm na cathrach amháin.",
        "rerank": "Cuir na sleachta iarrthóra in ord de réir ábharthachta don cheist.\n\n<Question> Cén sliocht a aithníonn an eochair chartlainne do {query}? </Question>\n\n{context}\n\nTabhair ar ais lipéid na sleachta amháin, ón gceann is fearr go dtí an ceann is measa, scartha le camóga.",
        "cite": "Freagair an cheist leis na sleachta agus luaigh lipéad an tsleachta tacaíochta.\n\n{context}\n\n<Question> Cad é an luach cláir do {query}? </Question>\nTabhair ar ais an luach agus lipéad lua i lúibíní cearnacha ina dhiaidh.",
        "longqa": "Léigh an doiciméad fada agus freagair an cheist.\n\n<document>\n{context}\n</document>\n\n<Question> Cén frása fíoraithe atá nasctha le {query}? </Question>\nFreagair leis an bhfrása amháin.",
        "summ": "Achoimrigh an doiciméad i dtrí phointe go díreach. Coinnigh na príomhainmneacha, na huimhreacha agus na torthaí.\n\n<document>\n{context}\n</document>",
        "icl": "Bain an mhapáil as na samplaí, ansin freagair an mhír dheireanach.\n\n{context}\n\nMír dheireanach: {query}\nFreagair leis an lipéad mapáilte amháin.",
    },
    "gl": {
        "recall": "Le o texto e lembra os datos de identificación.\n\n<text>\n{context}\n</text>\n\n<Question> Que código está asignado a {query}? </Question>\nResponde só co código.",
        "rag": "Usa as pasaxes recuperadas para responder a pregunta.\n\n{context}\n\n<Question> Que cidade acolleu a reunión sobre {query}? </Question>\nResponde só co nome da cidade.",
        "rerank": "Ordena as pasaxes candidatas por relevancia para a pregunta.\n\n<Question> Que pasaxe identifica a chave de arquivo de {query}? </Question>\n\n{context}\n\nDevolve só as etiquetas das pasaxes de mellor a peor, separadas por comas.",
        "cite": "Responde a pregunta usando as pasaxes e cita a etiqueta de apoio.\n\n{context}\n\n<Question> Cal é o valor de rexistro de {query}? </Question>\nDevolve o valor seguido da etiqueta de cita entre corchetes.",
        "longqa": "Le o documento longo e responde a pregunta.\n\n<document>\n{context}\n</document>\n\n<Question> Que frase de verificación está ligada a {query}? </Question>\nResponde só coa frase.",
        "summ": "Resume o documento en exactamente tres puntos. Conserva os nomes clave, números e resultados.\n\n<document>\n{context}\n</document>",
        "icl": "Deduce a correspondencia a partir dos exemplos e responde o elemento final.\n\n{context}\n\nElemento final: {query}\nResponde só coa etiqueta correspondente.",
    },
    "hr": {
        "recall": "Pročitajte tekst i zapamtite identifikacijske podatke.\n\n<text>\n{context}\n</text>\n\n<Question> Koji je kod dodijeljen za {query}? </Question>\nOdgovorite samo kodom.",
        "rag": "Upotrijebite dohvaćene odlomke za odgovor na pitanje.\n\n{context}\n\n<Question> Koji je grad bio domaćin sastanka o {query}? </Question>\nOdgovorite samo imenom grada.",
        "rerank": "Poredajte kandidatske odlomke prema relevantnosti za pitanje.\n\n<Question> Koji odlomak identificira arhivski ključ za {query}? </Question>\n\n{context}\n\nVratite samo oznake odlomaka od najbolje do najlošije, odvojene zarezima.",
        "cite": "Odgovorite na pitanje pomoću odlomaka i citirajte potpornu oznaku odlomka.\n\n{context}\n\n<Question> Koja je registarska vrijednost za {query}? </Question>\nVratite vrijednost praćenu oznakom citata u uglatim zagradama.",
        "longqa": "Pročitajte dugi dokument i odgovorite na pitanje.\n\n<document>\n{context}\n</document>\n\n<Question> Koja je verifikacijska fraza povezana s {query}? </Question>\nOdgovorite samo frazom.",
        "summ": "Sažmite dokument u točno tri točke. Sačuvajte ključna imena, brojeve i ishode.\n\n<document>\n{context}\n</document>",
        "icl": "Izvedite mapiranje iz primjera, zatim odgovorite na završnu stavku.\n\n{context}\n\nZavršna stavka: {query}\nOdgovorite samo mapiranom oznakom.",
    },
    "hu": {
        "recall": "Olvassa el a szöveget, és jegyezze meg az azonosító adatokat.\n\n<text>\n{context}\n</text>\n\n<Question> Milyen kód van hozzárendelve ehhez: {query}? </Question>\nCsak a kóddal válaszoljon.",
        "rag": "Használja a visszakeresett részleteket a kérdés megválaszolásához.\n\n{context}\n\n<Question> Melyik város adott otthont a(z) {query} témájú találkozónak? </Question>\nCsak a város nevével válaszoljon.",
        "rerank": "Rendezze a jelölt részleteket a kérdéshez való relevancia szerint.\n\n<Question> Melyik részlet azonosítja a(z) {query} archívumkulcsát? </Question>\n\n{context}\n\nCsak a részletcímkéket adja vissza a legjobbtól a legrosszabbig, vesszővel elválasztva.",
        "cite": "Válaszoljon a kérdésre a részletek alapján, és idézze a támogató részlet címkéjét.\n\n{context}\n\n<Question> Mi a(z) {query} regiszterértéke? </Question>\nAdja vissza az értéket, majd szögletes zárójelben az idézési címkét.",
        "longqa": "Olvassa el a hosszú dokumentumot, és válaszoljon a kérdésre.\n\n<document>\n{context}\n</document>\n\n<Question> Melyik ellenőrző kifejezés kapcsolódik ehhez: {query}? </Question>\nCsak a kifejezéssel válaszoljon.",
        "summ": "Foglalja össze a dokumentumot pontosan három pontban. Őrizze meg a kulcsneveket, számokat és eredményeket.\n\n<document>\n{context}\n</document>",
        "icl": "Következtesse ki a megfeleltetést a példákból, majd válaszoljon az utolsó elemre.\n\n{context}\n\nUtolsó elem: {query}\nCsak a hozzárendelt címkével válaszoljon.",
    },
    "is": {
        "recall": "Lestu textann og mundu auðkennisupplýsingarnar.\n\n<text>\n{context}\n</text>\n\n<Question> Hvaða kóða er úthlutað til {query}? </Question>\nSvaraðu aðeins með kóðanum.",
        "rag": "Notaðu sóttu textabrotin til að svara spurningunni.\n\n{context}\n\n<Question> Hvaða borg hélt fundinn um {query}? </Question>\nSvaraðu aðeins með nafni borgarinnar.",
        "rerank": "Raðaðu umsóknartextabrotunum eftir mikilvægi fyrir spurninguna.\n\n<Question> Hvaða textabrot auðkennir skjalasafnslykilinn fyrir {query}? </Question>\n\n{context}\n\nSkilaðu aðeins merkjum textabrotanna frá besta til versta, aðskildum með kommum.",
        "cite": "Svaraðu spurningunni með textabrotunum og vísaðu í stuðningsmerki textabrotsins.\n\n{context}\n\n<Question> Hvert er skráargildið fyrir {query}? </Question>\nSkilaðu gildinu og síðan tilvísunarmerkinu innan hornklofa.",
        "longqa": "Lestu langa skjalið og svaraðu spurningunni.\n\n<document>\n{context}\n</document>\n\n<Question> Hvaða staðfestingarsetning tengist {query}? </Question>\nSvaraðu aðeins með setningunni.",
        "summ": "Dragðu skjalið saman í nákvæmlega þrjá punkta. Haltu lykilheitum, tölum og niðurstöðum.\n\n<document>\n{context}\n</document>",
        "icl": "Ályktaðu vörpunina út frá dæmunum og svaraðu síðan síðasta atriðinu.\n\n{context}\n\nSíðasta atriði: {query}\nSvaraðu aðeins með vörpuðu merki.",
    },
    "lb": {
        "recall": "Liest den Text a behalt d'Identifikatiounsdonnéeën.\n\n<text>\n{context}\n</text>\n\n<Question> Wéi ee Code ass {query} zougewisen? </Question>\nÄntwert nëmme mam Code.",
        "rag": "Benotzt déi zeréckgesichte Passagen, fir d'Fro ze beäntweren.\n\n{context}\n\n<Question> Wéi eng Stad war Gaascht vum Treffen iwwer {query}? </Question>\nÄntwert nëmme mam Stadnumm.",
        "rerank": "Reit d'Kandidatepassagen no Relevanz fir d'Fro.\n\n<Question> Wéi eng Passage identifizéiert den Archivschlëssel fir {query}? </Question>\n\n{context}\n\nGitt nëmmen d'Passage-Etiketten zeréck, vun der beschter bis zur schlëmmster, mat Kommaen getrennt.",
        "cite": "Beäntwert d'Fro mat de Passagen a zitéiert d'ënnerstëtzend Passage-Etikett.\n\n{context}\n\n<Question> Wat ass de Registerwäert fir {query}? </Question>\nGitt de Wäert zeréck, gefollegt vun der Zitatiounsetikett an eckege Klameren.",
        "longqa": "Liest dat laangt Dokument a beäntwert d'Fro.\n\n<document>\n{context}\n</document>\n\n<Question> Wéi eng Verifizéierungsfras ass mat {query} verbonnen? </Question>\nÄntwert nëmme mat der Fras.",
        "summ": "Faasst d'Dokument a genee dräi Punkten zesummen. Behaalt d'Schlësselnimm, Zuelen an Resultater.\n\n<document>\n{context}\n</document>",
        "icl": "Leet d'Zouuerdnung aus de Beispiller of a beäntwert duerno dat lescht Element.\n\n{context}\n\nLescht Element: {query}\nÄntwert nëmme mat der zougeuerdneter Etikett.",
    },
    "lt": {
        "recall": "Perskaitykite tekstą ir įsiminkite identifikavimo duomenis.\n\n<text>\n{context}\n</text>\n\n<Question> Koks kodas priskirtas {query}? </Question>\nAtsakykite tik kodu.",
        "rag": "Naudokite rastas ištraukas atsakydami į klausimą.\n\n{context}\n\n<Question> Kuris miestas priėmė susitikimą apie {query}? </Question>\nAtsakykite tik miesto pavadinimu.",
        "rerank": "Surikiuokite kandidatines ištraukas pagal aktualumą klausimui.\n\n<Question> Kuri ištrauka identifikuoja archyvo raktą elementui {query}? </Question>\n\n{context}\n\nGrąžinkite tik ištraukų etiketes nuo geriausios iki blogiausios, atskirtas kableliais.",
        "cite": "Atsakykite į klausimą naudodami ištraukas ir pacituokite pagrindžiančios ištraukos etiketę.\n\n{context}\n\n<Question> Kokia yra registro reikšmė elementui {query}? </Question>\nGrąžinkite reikšmę, po jos pateikdami citatos etiketę laužtiniuose skliaustuose.",
        "longqa": "Perskaitykite ilgą dokumentą ir atsakykite į klausimą.\n\n<document>\n{context}\n</document>\n\n<Question> Kokia patvirtinimo frazė susieta su {query}? </Question>\nAtsakykite tik fraze.",
        "summ": "Apibendrinkite dokumentą tiksliai trimis punktais. Išsaugokite pagrindinius pavadinimus, skaičius ir rezultatus.\n\n<document>\n{context}\n</document>",
        "icl": "Iš pavyzdžių nustatykite atitikimą, tada atsakykite į paskutinį elementą.\n\n{context}\n\nPaskutinis elementas: {query}\nAtsakykite tik priskirta etikete.",
    },
    "lv": {
        "recall": "Izlasiet tekstu un atcerieties identifikācijas datus.\n\n<text>\n{context}\n</text>\n\n<Question> Kāds kods ir piešķirts {query}? </Question>\nAtbildiet tikai ar kodu.",
        "rag": "Izmantojiet izgūtos fragmentus, lai atbildētu uz jautājumu.\n\n{context}\n\n<Question> Kura pilsēta rīkoja sanāksmi par {query}? </Question>\nAtbildiet tikai ar pilsētas nosaukumu.",
        "rerank": "Sakārtojiet kandidātfragmentus pēc atbilstības jautājumam.\n\n<Question> Kurš fragments identificē arhīva atslēgu priekš {query}? </Question>\n\n{context}\n\nAtgrieziet tikai fragmentu etiķetes no labākās līdz sliktākajai, atdalītas ar komatiem.",
        "cite": "Atbildiet uz jautājumu, izmantojot fragmentus, un citējiet atbalstošā fragmenta etiķeti.\n\n{context}\n\n<Question> Kāda ir reģistra vērtība priekš {query}? </Question>\nAtgrieziet vērtību, kam seko citēšanas etiķete kvadrātiekavās.",
        "longqa": "Izlasiet garo dokumentu un atbildiet uz jautājumu.\n\n<document>\n{context}\n</document>\n\n<Question> Kāda verifikācijas frāze ir saistīta ar {query}? </Question>\nAtbildiet tikai ar frāzi.",
        "summ": "Apkopojiet dokumentu tieši trīs punktos. Saglabājiet atslēgvārdus, skaitļus un rezultātus.\n\n<document>\n{context}\n</document>",
        "icl": "Izseciniet kartējumu no piemēriem un pēc tam atbildiet uz pēdējo vienumu.\n\n{context}\n\nPēdējais vienums: {query}\nAtbildiet tikai ar kartēto etiķeti.",
    },
    "mk": {
        "recall": "Прочитајте го текстот и запомнете ги идентификациските податоци.\n\n<text>\n{context}\n</text>\n\n<Question> Кој код е доделен на {query}? </Question>\nОдговорете само со кодот.",
        "rag": "Користете ги пронајдените пасуси за да одговорите на прашањето.\n\n{context}\n\n<Question> Кој град беше домаќин на состанокот за {query}? </Question>\nОдговорете само со името на градот.",
        "rerank": "Подредете ги кандидатските пасуси според релевантноста за прашањето.\n\n<Question> Кој пасус го идентификува архивскиот клуч за {query}? </Question>\n\n{context}\n\nВратете ги само етикетите на пасусите од најдобар до најлош, одделени со запирки.",
        "cite": "Одговорете на прашањето користејќи ги пасусите и цитирајте ја поддржувачката етикета на пасусот.\n\n{context}\n\n<Question> Која е регистарската вредност за {query}? </Question>\nВратете ја вредноста проследена со етикетата за цитирање во квадратни загради.",
        "longqa": "Прочитајте го долгиот документ и одговорете на прашањето.\n\n<document>\n{context}\n</document>\n\n<Question> Која проверочна фраза е поврзана со {query}? </Question>\nОдговорете само со фразата.",
        "summ": "Сумирајте го документот во точно три точки. Задржете ги клучните имиња, бројките и исходите.\n\n<document>\n{context}\n</document>",
        "icl": "Изведете го мапирањето од примерите, потоа одговорете на последната ставка.\n\n{context}\n\nПоследна ставка: {query}\nОдговорете само со мапираната етикета.",
    },
    "mt": {
        "recall": "Aqra t-test u ftakar fid-data ta' identifikazzjoni.\n\n<text>\n{context}\n</text>\n\n<Question> Liema kodiċi ġie assenjat lil {query}? </Question>\nWieġeb biss bil-kodiċi.",
        "rag": "Uża s-siltiet miġbura biex twieġeb il-mistoqsija.\n\n{context}\n\n<Question> Liema belt ospitat il-laqgħa dwar {query}? </Question>\nWieġeb biss bl-isem tal-belt.",
        "rerank": "Ikklassifika s-siltiet kandidati skont ir-rilevanza għall-mistoqsija.\n\n<Question> Liema silta tidentifika ċ-ċavetta tal-arkivju għal {query}? </Question>\n\n{context}\n\nIrritorna biss it-tikketti tas-siltiet mill-aħjar għall-agħar, separati b'virgoli.",
        "cite": "Wieġeb il-mistoqsija billi tuża s-siltiet u ċċita t-tikketta tas-silta ta' appoġġ.\n\n{context}\n\n<Question> X'inhu l-valur tar-reġistru għal {query}? </Question>\nIrritorna l-valur segwit mit-tikketta taċ-ċitazzjoni fil-parentesi kwadri.",
        "longqa": "Aqra d-dokument twil u wieġeb il-mistoqsija.\n\n<document>\n{context}\n</document>\n\n<Question> Liema frażi ta' verifika hija marbuta ma' {query}? </Question>\nWieġeb biss bil-frażi.",
        "summ": "Agħmel sommarju tad-dokument fi tliet punti eżatti. Żomm l-ismijiet ewlenin, in-numri u r-riżultati.\n\n<document>\n{context}\n</document>",
        "icl": "Inferixxi l-immappjar mill-eżempji, imbagħad wieġeb l-aħħar oġġett.\n\n{context}\n\nL-aħħar oġġett: {query}\nWieġeb biss bit-tikketta mmappjata.",
    },
    "sq": {
        "recall": "Lexoni tekstin dhe mbani mend të dhënat identifikuese.\n\n<text>\n{context}\n</text>\n\n<Question> Cili kod i është caktuar {query}? </Question>\nPërgjigjuni vetëm me kodin.",
        "rag": "Përdorni fragmentet e gjetura për t'iu përgjigjur pyetjes.\n\n{context}\n\n<Question> Cili qytet priti takimin për {query}? </Question>\nPërgjigjuni vetëm me emrin e qytetit.",
        "rerank": "Renditni fragmentet kandidate sipas rëndësisë për pyetjen.\n\n<Question> Cili fragment identifikon çelësin e arkivit për {query}? </Question>\n\n{context}\n\nKtheni vetëm etiketat e fragmenteve nga më e mira te më e dobëta, të ndara me presje.",
        "cite": "Përgjigjuni pyetjes duke përdorur fragmentet dhe citoni etiketën mbështetëse të fragmentit.\n\n{context}\n\n<Question> Cila është vlera e regjistrit për {query}? </Question>\nKtheni vlerën të ndjekur nga etiketa e citimit në kllapa katrore.",
        "longqa": "Lexoni dokumentin e gjatë dhe përgjigjuni pyetjes.\n\n<document>\n{context}\n</document>\n\n<Question> Cila frazë verifikimi lidhet me {query}? </Question>\nPërgjigjuni vetëm me frazën.",
        "summ": "Përmblidheni dokumentin në saktësisht tre pika. Ruani emrat kyç, numrat dhe rezultatet.\n\n<document>\n{context}\n</document>",
        "icl": "Nxirrni hartëzimin nga shembujt dhe pastaj përgjigjuni elementit të fundit.\n\n{context}\n\nElementi i fundit: {query}\nPërgjigjuni vetëm me etiketën e hartëzuar.",
    },
    "sr": {
        "recall": "Прочитајте текст и запамтите идентификационе податке.\n\n<text>\n{context}\n</text>\n\n<Question> Који код је додељен за {query}? </Question>\nОдговорите само кодом.",
        "rag": "Користите пронађене одломке да одговорите на питање.\n\n{context}\n\n<Question> Који град је био домаћин састанка о {query}? </Question>\nОдговорите само именом града.",
        "rerank": "Поређајте кандидатске одломке по релевантности за питање.\n\n<Question> Који одломак идентификује архивски кључ за {query}? </Question>\n\n{context}\n\nВратите само ознаке одломака од најбоље до најлошије, одвојене зарезима.",
        "cite": "Одговорите на питање користећи одломке и цитирајте подржавајућу ознаку одломка.\n\n{context}\n\n<Question> Која је регистарска вредност за {query}? </Question>\nВратите вредност праћену ознаком цитата у угластим заградама.",
        "longqa": "Прочитајте дугачак документ и одговорите на питање.\n\n<document>\n{context}\n</document>\n\n<Question> Која је верификациона фраза повезана са {query}? </Question>\nОдговорите само фразом.",
        "summ": "Сажмите документ у тачно три ставке. Сачувајте кључна имена, бројеве и исходе.\n\n<document>\n{context}\n</document>",
        "icl": "Изведите мапирање из примера, затим одговорите на последњу ставку.\n\n{context}\n\nПоследња ставка: {query}\nОдговорите само мапираном ознаком.",
    },
    "tr": {
        "recall": "Metni okuyun ve tanımlama bilgilerini hatırlayın.\n\n<text>\n{context}\n</text>\n\n<Question> {query} için hangi kod atanmıştır? </Question>\nYalnızca kodla yanıtlayın.",
        "rag": "Soruyu yanıtlamak için getirilen pasajları kullanın.\n\n{context}\n\n<Question> {query} toplantısına hangi şehir ev sahipliği yaptı? </Question>\nYalnızca şehir adıyla yanıtlayın.",
        "rerank": "Aday pasajları soruya olan ilgilerine göre sıralayın.\n\n<Question> Hangi pasaj {query} için arşiv anahtarını tanımlar? </Question>\n\n{context}\n\nYalnızca pasaj etiketlerini en iyiden en kötüye doğru, virgülle ayırarak döndürün.",
        "cite": "Pasajları kullanarak soruyu yanıtlayın ve destekleyici pasaj etiketini alıntılayın.\n\n{context}\n\n<Question> {query} için kayıt değeri nedir? </Question>\nDeğeri ve ardından köşeli parantez içinde alıntı etiketini döndürün.",
        "longqa": "Uzun belgeyi okuyun ve soruyu yanıtlayın.\n\n<document>\n{context}\n</document>\n\n<Question> {query} ile hangi doğrulama ifadesi bağlantılıdır? </Question>\nYalnızca ifadeyle yanıtlayın.",
        "summ": "Belgeyi tam olarak üç madde halinde özetleyin. Anahtar adları, sayıları ve sonuçları koruyun.\n\n<document>\n{context}\n</document>",
        "icl": "Örneklerden eşlemeyi çıkarın, ardından son öğeyi yanıtlayın.\n\n{context}\n\nSon öğe: {query}\nYalnızca eşlenen etiketle yanıtlayın.",
    },
}


REFERENCE_PACKS = {
    "bg": ("архив, договор, карта, доклад, мост, река, училище, пазар, музей, станция, градина, езеро", "София, Пловдив, Варна, Бургас, Русе, Велико Търново", "одобрено, отложено, завършено, проверено"),
    "hr": ("arhiv, ugovor, karta, izvještaj, most, rijeka, škola, tržnica, muzej, stanica, vrt, jezero", "Zagreb, Split, Rijeka, Osijek, Zadar, Pula", "odobreno, odgođeno, dovršeno, provjereno"),
    "cs": ("archiv, smlouva, mapa, zpráva, most, řeka, škola, trh, muzeum, stanice, zahrada, jezero", "Praha, Brno, Ostrava, Plzeň, Liberec, Olomouc", "schváleno, odloženo, dokončeno, ověřeno"),
    "da": ("arkiv, kontrakt, kort, rapport, bro, flod, skole, marked, museum, station, have, sø", "København, Aarhus, Odense, Aalborg, Esbjerg, Roskilde", "godkendt, udskudt, afsluttet, verificeret"),
    "nl": ("archief, contract, kaart, rapport, brug, rivier, school, markt, museum, station, tuin, meer", "Amsterdam, Rotterdam, Utrecht, Den Haag, Eindhoven, Groningen", "goedgekeurd, uitgesteld, voltooid, geverifieerd"),
    "et": ("arhiiv, leping, kaart, aruanne, sild, jõgi, kool, turg, muuseum, jaam, aed, järv", "Tallinn, Tartu, Narva, Pärnu, Viljandi, Rakvere", "heaks kiidetud, edasi lükatud, lõpetatud, kontrollitud"),
    "fi": ("arkisto, sopimus, kartta, raportti, silta, joki, koulu, tori, museo, asema, puutarha, järvi", "Helsinki, Tampere, Turku, Oulu, Espoo, Kuopio", "hyväksytty, lykätty, valmistunut, vahvistettu"),
    "fr": ("archive, contrat, carte, rapport, pont, rivière, école, marché, musée, gare, jardin, lac", "Paris, Lyon, Marseille, Lille, Nantes, Toulouse", "approuvé, reporté, terminé, vérifié"),
    "de": ("Archiv, Vertrag, Karte, Bericht, Brücke, Fluss, Schule, Markt, Museum, Bahnhof, Garten, See", "Berlin, Hamburg, München, Köln, Frankfurt, Leipzig", "genehmigt, verschoben, abgeschlossen, geprüft"),
    "el": ("αρχείο, σύμβαση, χάρτης, αναφορά, γέφυρα, ποτάμι, σχολείο, αγορά, μουσείο, σταθμός, κήπος, λίμνη", "Αθήνα, Θεσσαλονίκη, Πάτρα, Ηράκλειο, Λάρισα, Βόλος", "εγκρίθηκε, αναβλήθηκε, ολοκληρώθηκε, επαληθεύτηκε"),
    "hu": ("archívum, szerződés, térkép, jelentés, híd, folyó, iskola, piac, múzeum, állomás, kert, tó", "Budapest, Debrecen, Szeged, Pécs, Győr, Miskolc", "jóváhagyva, elhalasztva, befejezve, ellenőrizve"),
    "ga": ("cartlann, conradh, mapa, tuarascáil, droichead, abhainn, scoil, margadh, músaem, stáisiún, gairdín, loch", "Baile Átha Cliath, Corcaigh, Gaillimh, Luimneach, Port Láirge, Sligeach", "ceadaithe, curtha siar, críochnaithe, fíoraithe"),
    "it": ("archivio, contratto, mappa, rapporto, ponte, fiume, scuola, mercato, museo, stazione, giardino, lago", "Roma, Milano, Torino, Napoli, Bologna, Firenze", "approvato, rinviato, completato, verificato"),
    "lv": ("arhīvs, līgums, karte, ziņojums, tilts, upe, skola, tirgus, muzejs, stacija, dārzs, ezers", "Rīga, Daugavpils, Liepāja, Jelgava, Valmiera, Ventspils", "apstiprināts, atlikts, pabeigts, pārbaudīts"),
    "lt": ("archyvas, sutartis, žemėlapis, ataskaita, tiltas, upė, mokykla, turgus, muziejus, stotis, sodas, ežeras", "Vilnius, Kaunas, Klaipėda, Šiauliai, Panevėžys, Alytus", "patvirtinta, atidėta, užbaigta, patikrinta"),
    "mt": ("arkivju, kuntratt, mappa, rapport, pont, xmara, skola, suq, mużew, stazzjon, ġnien, lag", "Valletta, Mdina, Birkirkara, Sliema, Mosta, Rabat", "approvat, pospost, komplut, verifikat"),
    "pl": ("archiwum, umowa, mapa, raport, most, rzeka, szkoła, rynek, muzeum, stacja, ogród, jezioro", "Warszawa, Kraków, Gdańsk, Wrocław, Poznań, Łódź", "zatwierdzone, odłożone, zakończone, zweryfikowane"),
    "pt": ("arquivo, contrato, mapa, relatório, ponte, rio, escola, mercado, museu, estação, jardim, lago", "Lisboa, Porto, Coimbra, Braga, Faro, Aveiro", "aprovado, adiado, concluído, verificado"),
    "ro": ("arhivă, contract, hartă, raport, pod, râu, școală, piață, muzeu, stație, grădină, lac", "București, Cluj, Iași, Timișoara, Brașov, Constanța", "aprobat, amânat, finalizat, verificat"),
    "sk": ("archív, zmluva, mapa, správa, most, rieka, škola, trh, múzeum, stanica, záhrada, jazero", "Bratislava, Košice, Žilina, Nitra, Prešov, Trnava", "schválené, odložené, dokončené, overené"),
    "sl": ("arhiv, pogodba, zemljevid, poročilo, most, reka, šola, trg, muzej, postaja, vrt, jezero", "Ljubljana, Maribor, Celje, Koper, Kranj, Novo mesto", "odobreno, preloženo, končano, preverjeno"),
    "es": ("archivo, contrato, mapa, informe, puente, río, escuela, mercado, museo, estación, jardín, lago", "Madrid, Barcelona, Valencia, Sevilla, Bilbao, Granada", "aprobado, aplazado, completado, verificado"),
    "sv": ("arkiv, avtal, karta, rapport, bro, flod, skola, marknad, museum, station, trädgård, sjö", "Stockholm, Göteborg, Malmö, Uppsala, Lund, Umeå", "godkänt, uppskjutet, slutfört, verifierat"),
    "en": ("archive, contract, map, report, bridge, river, school, market, museum, station, garden, lake", "London, Dublin, Edinburgh, Cardiff, Manchester, Bristol", "approved, delayed, completed, verified"),
    "sq": ("arkiv, kontratë, hartë, raport, urë, lumë, shkollë, treg, muze, stacion, kopsht, liqen", "Tiranë, Durrës, Shkodër, Vlorë, Korçë, Elbasan", "miratuar, shtyrë, përfunduar, verifikuar"),
    "eu": ("artxibo, kontratu, mapa, txosten, zubi, ibai, eskola, merkatu, museo, geltoki, lorategi, aintzira", "Bilbo, Donostia, Gasteiz, Iruñea, Baiona, Eibar", "onartua, atzeratua, amaitua, egiaztatua"),
    "bs": ("arhiv, ugovor, karta, izvještaj, most, rijeka, škola, pijaca, muzej, stanica, bašta, jezero", "Sarajevo, Banja Luka, Mostar, Tuzla, Zenica, Bihać", "odobreno, odgođeno, završeno, provjereno"),
    "ca": ("arxiu, contracte, mapa, informe, pont, riu, escola, mercat, museu, estació, jardí, llac", "Barcelona, Girona, Tarragona, Lleida, València, Palma", "aprovat, ajornat, completat, verificat"),
    "gl": ("arquivo, contrato, mapa, informe, ponte, río, escola, mercado, museo, estación, xardín, lago", "Santiago, Vigo, A Coruña, Lugo, Ourense, Pontevedra", "aprobado, adiado, completado, verificado"),
    "is": ("skjalasafn, samningur, kort, skýrsla, brú, á, skóli, markaður, safn, stöð, garður, vatn", "Reykjavík, Akureyri, Keflavík, Selfoss, Ísafjörður, Egilsstaðir", "samþykkt, frestað, lokið, staðfest"),
    "lb": ("Archiv, Kontrakt, Kaart, Rapport, Bréck, Floss, Schoul, Maart, Musée, Gare, Gaart, Séi", "Lëtzebuerg, Esch, Diddeleng, Déifferdeng, Ettelbréck, Miersch", "genehmegt, verréckelt, ofgeschloss, iwwerpréift"),
    "mk": ("архива, договор, карта, извештај, мост, река, училиште, пазар, музеј, станица, градина, езеро", "Скопје, Битола, Охрид, Тетово, Куманово, Прилеп", "одобрено, одложено, завршено, проверено"),
    "no": ("arkiv, kontrakt, kart, rapport, bro, elv, skole, marked, museum, stasjon, hage, innsjø", "Oslo, Bergen, Trondheim, Stavanger, Tromsø, Bodø", "godkjent, utsatt, fullført, verifisert"),
    "ru": ("архив, договор, карта, отчёт, мост, река, школа, рынок, музей, станция, сад, озеро", "Москва, Санкт-Петербург, Казань, Новосибирск, Екатеринбург, Самара", "одобрено, отложено, завершено, проверено"),
    "sr": ("архива, уговор, карта, извештај, мост, река, школа, пијаца, музеј, станица, башта, језеро", "Београд, Нови Сад, Ниш, Крагујевац, Суботица, Зрењанин", "одобрено, одложено, завршено, проверено"),
    "tr": ("arşiv, sözleşme, harita, rapor, köprü, nehir, okul, pazar, müze, istasyon, bahçe, göl", "İstanbul, Ankara, İzmir, Bursa, Konya, Antalya", "onaylandı, ertelendi, tamamlandı, doğrulandı"),
    "uk": ("архів, договір, карта, звіт, міст, річка, школа, ринок, музей, станція, сад, озеро", "Київ, Львів, Одеса, Харків, Дніпро, Чернівці", "схвалено, відкладено, завершено, перевірено"),
    "cy": ("archif, contract, map, adroddiad, pont, afon, ysgol, marchnad, amgueddfa, gorsaf, gardd, llyn", "Caerdydd, Abertawe, Wrecsam, Bangor, Casnewydd, Aberystwyth", "cymeradwywyd, gohiriwyd, cwblhawyd, dilyswyd"),
}


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def reference_pack(lang: str) -> dict[str, list[str]]:
    terms, cities, outcomes = REFERENCE_PACKS[lang]
    return {
        "terms": _split(terms),
        "cities": _split(cities),
        "outcomes": _split(outcomes),
    }


def prompt_for(lang: str, task: str) -> str:
    return PROMPTS.get(lang, BASE_PROMPTS)[task]

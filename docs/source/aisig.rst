============
Introduktion
============
Kodebasen som er dokumenteret her er tilvejebragt ifm. signaturprojektet "AI Aktindsigt" som har haft til formål at undersøge anvendelsen af kunstig intelligens til at optimere behandlingen af aktindsigtssager i den offentlige sektor.

Det overordnede formål med projektet har været at imødekomme især kommunale interesser ved at udvikle og udstille bredt favnende open-source (OS) resourcer, herunder specifikt to "natural language processing" NLP modeller til hhv. assisteret anonymisering af personhenførbare oplysninger i aktindsigtsrelevante dokumenter og semantisk søgning i det typisk omfattende kildematerialet som ligger til grund for en aktindsigtssag.

Konkret har projektet bidraget med følgende resourcer:
    * Et nyt dansk datasæt bestående af n unikke sætninger* som er skrabet fra 94 af de 98 kommuners hjemmesider. En delmængde af datasættet er desuden annoteret med KL-område til brug i træningen af en sætnings-klassifikations model.
    * En ny dansk sprogmodel trænet på ovenstående datasæt.
    * En ny kommunale sprogmodel trænet med differential privacy på aktindsigtsdata fra Sønderborg Kommune.
    * En ny kommunal named entity recognition (NER) model trænet med differential privacy på annoterede aktindsigtsdata fra Sønderborg Kommune.
    * Nærværende kodebase til træning af sprogmodellerne med og uden differential privacy.

Vedr. den tværkommunale sprogmodel
----------------------------------------------
**Mangler**

Vedr. den kommunal sprogmodel model til assisteret anonymisering af personhenførbart data
-----------------------------------------------------------------------------------------
**Mangler**

Vedr. den kommunal sprogmodel til semantisk søgning
---------------------------------------------------
**Mangler**

Vedr. Differential privacy og open-sourcing
-------------------------------------------
I projektet har vi anvendt maskinlæring som er en subgenre af statistisk modellering til at træne sprogmodellerne. Groft set kan maskinlæringsmodeller anskues som datadrevne problemløsere der er trænet med generelle metoder til at finde mønstre i data.

Selve træningen af en maskinlæringsmodel involvere to fundamentale elmeneter af tilfældighed:
    1. Det træningsdata som bruges til træningen af en maskinlæringsmodel er typisk en afgrænset og tilfældig stikprøve fra en interessant datakilde.
    2. De metoder som anvendes til at træne en maskinlæringsmodel er oftest baseret på randomiserede optimerings algoritmer for at øge træningens fokus på robust mønstergenkendelse.

Dette medfører at enhver maskinlæringsmodels "viden" er forbundet med en naturlig usikkerhed som spreder sig til al anvendelse af modellen. Eksempelvis er det usikkert om en NER model har ret når den anvendes til at vurdere om et givet ord er en personhenførbar information eller ej - modellen kan enten gætte rigtigt eller forkert, og chancen for at den gætter rigtigt er direkte afhængig af den "viden" den har. Fuldstændig som det gør sig gældende med et menneske.

Det er muligt at danne et upartisk billede af en maskinlæringsmodels evne til at løse et konkret problem ved at "teste" modellen på en ny uafhængig stikprøve data som er hentet fra samme datakilde som træningsdataet, og derfor forventeligt deler de samme mønstre. Det kan anskues lidt som når vi tester en studerendes evner til en eksamen; metoderne vi vil have den studerende til at anvende er de samme som hun er blevet trænet i at bruge i undervisningen, men opgaveformuleringen til eksamen skal være ny før vi kan vurdere om den studerende reelt har lært at anvende fagets metoder eller blot har lært at recitere. Når vi tester en maskinlæringsmodels evner, får vi på samme måde som med en studerende til en eksamen, et øjebliks billede af hvor god modellen er til at løse et givent opgavesæt. Sætter vi modellen til at løse flere opgavesæt kan vi danne os et billede af modellens gennemsnitslige evne til at løse det konkrete problem vi har trænet den til at løse, ligesom vi kan dannes os et billede af hvor den har svagheder og styrker samt hvor stort et udsving i antal korrekt løste opgaver vi kan forvente hvis vi sætter modellen til at løse et nyt opgavesæt.

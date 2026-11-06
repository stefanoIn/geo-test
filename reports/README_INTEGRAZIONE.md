# GeoTest — Moduli + pipeline YAML

Questi file vanno **copiati nella radice di `D:\UNI\GeoTest`**, conservando la struttura delle cartelle e sovrascrivendo i moduli omonimi. Non è stata eseguita alcuna modifica automatica su GitHub.

## Novità

- `src/models/factory.py`: costruisce `CROMA` o `TerraMind`, `TokenToSpatial` e `FCNHead` partendo dal YAML. Non gestisce dati, training o normalizzazione.
- `src/evaluation/sliding_window.py`: supporta `stride` esplicito, ad esempio `tile_size=120`, `stride=60`. Senza stride mantiene la precedente distribuzione uniforme delle finestre.
- `src/evaluation/evaluator.py` e `metrics.py`: evaluation generica per segmentazione e DataLoader, con matrice di confusione, TP/FP/FN/TN, pixel corretti/errati, pixel ignorati, IoU, F1, precision, recall, accuracy, loss e risultati per campione.
- `training/checkpoints.py`: salvataggio/caricamento completo o per componenti, opzionalmente con optimizer e scheduler.
- `training/trainer.py`: singola epoca di training; accetta i nomi degli input (`input_keys`) per non inoltrare metadati al modello.
- `src/data/datasets/sen1floods11.py`: aggiunge la chiave **`chip_id`** a ogni sample per collegare i conteggi al chip reale. Il modulo di evaluation resta del tutto indipendente da Sen1Floods11.
- `src/encoders/terramind/terramind.py`: stesso wrapper multimodale generalizzato già validato nei test S1, S2, S1+S2.
- `run.py`: unico orchestratore; preleva impostazioni dal YAML, non replica la logica dei moduli.
- YAML completi per CROMA S1+S2, TerraMind S1+S2 e TerraMind S1.

## Esecuzione (PowerShell, dalla root del repository)

Ispezione dati e statistiche, senza caricare i foundation model:

```powershell
python run.py --mode inspect --config configs/croma/croma_s1s2_fcn_all_hand_labels.yaml
```

Training CROMA S1+S2 (si consiglia **prima** di ridurre a 1-2 epoche per uno smoke test reale):

```powershell
python run.py --mode train --config configs/croma/croma_s1s2_fcn_all_hand_labels.yaml
```

Training TerraMind S1+S2:

```powershell
python run.py --mode train --config configs/terramind/terramind_v1_large_s1s2_fcn_all_hand_labels.yaml
```

Training TerraMind solo SAR:

```powershell
python run.py --mode train --config configs/terramind/terramind_v1_large_s1_fcn_all_hand_labels.yaml
```

Valutazione **separata** del checkpoint selezionato su test:

```powershell
python run.py --mode evaluate --config configs/croma/croma_s1s2_fcn_all_hand_labels.yaml --checkpoint runs/sen1floods11/croma/fcn_s1s2/RUN_ID/best.pt --split test
```

Per Bolivia: `--split bolivia`. Per specificare un percorso del report: `--report runs/.../bolivia_report.json`.

Per riprendere un run interrotto dal checkpoint più recente:

```powershell
python run.py --mode train --config configs/croma/croma_s1s2_fcn_all_hand_labels.yaml --resume runs/sen1floods11/croma/fcn_s1s2/RUN_ID/latest.pt
```

**Attenzione:** il resume ripristina pesi, optimizer e scheduler, **non** lo stato completo dei generatori casuali/DataLoader; quindi non garantisce una ripresa bit-per-bit identica. Non cambiare il YAML durante un resume.

## File dei risultati

```text
runs/.../RUN_ID/
    config.yaml             # snapshot della configurazione
    metadata.json           # split CSV SHA256, pretrained checkpoint path/size, versione PyTorch
    history.json            # training loss e VAL metrics/conteggi per ogni epoca
    best.pt                 # miglior modello secondo la metrica sulla VAL
    latest.pt               # pesi + optimizer + scheduler per resume
    best_validation.json    # VAL dettagliata, inclusi TP/FP/FN/TN per chip_id
    evaluation_test.json    # prodotto SOLO dal comando --mode evaluate
    evaluation_bolivia.json # prodotto SOLO dal comando --mode evaluate
```

Con encoder congelati, `best.pt` e `latest.pt` salvano `adapter` e `head`, **non** duplicano i pesi pretrained; per caricarli devi mantenere il checkpoint pretrained originale. Se l'encoder non è congelato, viene salvato l'intero modello.

## Protocollo e limiti attuali

- Il runner attuale costruisce dataset Sen1Floods11 (è l'unico adapter di dataset implementato); **factory, trainer, evaluator, checkpoint e sliding-window sono indipendenti dal dataset**.
- I YAML impostano `class_weights: [1.0, 8.0]`, selezione del best checkpoint in base a **IoU della classe 1 (water)**, e overlapping tiles 120/60, in linea con i notebook di prova. Questa **modifica la precedente configurazione CROMA**, che selezionava `miou`.
- In validation e test si usa il medesimo normalizzatore derivato dal **training split**, non dalle immagini da valutare.
- Le normalizzazioni `two_sigma_stretch` scelte sono una configurazione sperimentale, **non equivalgono necessariamente al preprocessing nativo del pretraining** di TerraMind e CROMA; per il confronto scientifico definitivo sarà necessario dichiarare e verificare il protocollo.
- I CSV 10/25/100% vanno indicati nei YAML: le statistiche training sono risolte nuovamente in base al contenuto del CSV. Il runner non esegue automaticamente una griglia di esperimenti.
- `evaluator.py` valuta segmentazione e richiede logits `[B,C,H,W]`, target `[B,H,W]` e un DataLoader compatibile; non è un evaluator universale per classificazione/regressione.
- Gli ID dei campioni e le etichette possono essere conservati, ma non si calcolano in questo step metriche *per regione geografica*: queste richiedono metadati di dominio/split affidabili.

## Verifiche

Sintassi dei file compilata, YAML validati e smoke test end-to-end **su CPU con encoder sintetici** per due configurazioni: costruzione modelli, training, validation, checkpoint, reload, test, report JSON e chip_id. **Non** è un test end-to-end dei checkpoint reali su Sen1Floods11: verifica preliminare necessaria sul computer con GPU.

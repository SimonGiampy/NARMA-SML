## NARMA - SML

Narma project


temporal dependence data with binary classification from 5 features, 1 binary label, 20000 length

data generator river -> datasets -> synth

implement arima or narma model of high order

create 2 or 3 similar datasets

temporal dependence between past labels and some features

create concept drift in the mechanism for assigning labels or in function generator

evaluate:
- temporal correlation in every feature x and label y, through features and labels
- temporal correlation for every feature x and lable y in past data
- feature significativity in prediction
- plots pacf and acf with their analysis and description

SML models:
- HAT, ARF models (with concept drift detectors)
- HAT, ARF models with the addition of temporal augmentation
- kappa statistics
- kappa temporal (in kappa_t)
- models accuracy and the other metrics

temporal augmentation = add past labels in SML classification with different order

apply prequential evaluation with SML models


Professor notes:

Di seguito trovi l’elenco riassuntivo per il progetto:

- Creazione di dati con dipendenza temporale, utilizza i processi tipici delle time series ARMA, NARMA, ARIMA (sono sufficienti 5 features) per un problema di classificazione binaria
- Deve esserci dipendenza temporale tra le singole features (ogni feature dev'essere una time series) non è richiesta correlazione tra features diverse, ma volendo puoi sperimentare
- Deve esserci una correlazione significativa tra l'etichetta y attuale e le etichette y nel passato, e anche tra l'etichetta y attuale e le feature nel passato (esempio di funzione y = moda(....))
- Per ogni dataset creato (2/3) inserisci almeno un concept drift, puoi ispirarti a come vengono creati i drift nei dati SEA su river (cambia il modo di combinare le features oppure l’ordine all’interno dei processi ARIMA/NARMA di generazione dati)
  
- Presenta i plot della ACF/PACF delle singole features, la correlazione tra le y e le y passate e feature passate (puoi utilizzare anche altri metodi per valutare la significatività di una feature per la label)
- Testa i dati creati comparando HAT, ARF con i rispettivi modelli con la Temporal Augmentation (scenario di classificazione come visto a lezione) 
- Utilizza come metriche di comparazione la Kappa Statistics e la Kappa Temporal

Ti allego poi il codice del TemporallyAugmentedClassifier, lo usi come i modelli di classificazione di river. Gli unici due parametri sono: base_learner, che rappresenta il modello su cui fare l’augmentation (HAT o ARF), e num_old_labels che indica quante label nel passato considerare nell’augmentation. Infine trovi il codice della Kappa Temporal da usare come metrica aggiuntiva alla Kappa Statistics nella valutazione finale.

1. https://ollama.com
2. terminalu: ollama run codellama:7b-instruct - uruchomienie modelu
3. scg-cli zmodyfikowany: https://github.com/jciura/scg-cli-modified


4. scg-cli generate <Sciezka> - najpierw generuje się zawsze dla projektu pojedyncze grafy semantyczne, bez tego reszata
   nie dziala
5. scg-cli export -g CCN -o gdf <Sciezka> - export całego grafu do pliku .gdf
6. scg-cli summary -g SCG <Sciezka> - szybkie podsumiwanie projektu
7. scg-cli crucial <Sciezka> -n k; k - ile bierzemy węzłów o najwyższych wartościach, raczej chcemy podawać all, żeby
   każdy embedding miał te wartości, a nie tylko wybrane
8. scg-cli partition n; podaje jak podzielić projekt na n partycji
9. Gephi - plik do otwierania plików .gdf - można użyć ale raczej bezużyteczny - nie na tym nie widać
10. Przy zmianie scg-cli wywołać: sbt clean universal:packageBin, żeby wygenerować nową paczkę
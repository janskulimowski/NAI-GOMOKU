<p align="center">
<img src="https://github.com/janskulimowski/NAI-GOMOKU/blob/main/gomoku_logo.png?raw=true" width="400" alt="Gomoku" />
</p>

## Opis pro­jektu

Pro­jekt polega na imple­men­ta­cji turo­wej, deter­mi­ni­stycz­nej gry dwu­oso­bo­wej o sumie zero­wej, w któ­rej dwóch gra­czy rywa­li­zuje, aby uło­żyć pięć swo­ich pion­ków w rzę­dzie (poziomo, pio­nowo lub na ukos) na plan­szy **10x10**. Gra jest odmianą kla­sycz­nego **Kółko i Krzy­żyk**, ale z więk­szą plan­szą i celem.

W ramach zada­nia, poza samą grą, zaim­ple­men­to­wano rów­nież sztuczną inte­li­gen­cję (**SI**), która potrafi grać prze­ciwko czło­wie­kowi, wyko­rzy­stu­jąc biblio­tekę **EasyAI** i algo­rytm **Nega­max**.

## Wyma­ga­nia wstępne

Pro­jekt został zaim­ple­men­to­wany w języku Python. Wymaga zain­sta­lo­wa­nia biblio­teki EasyAI.

Możesz zain­sta­lo­wać wyma­gane pakiety, uży­wa­jąc poniż­szego pole­ce­nia:


    pip install easyAI

## Uru­cho­mie­nie

Aby uru­cho­mić grę, wyko­naj nastę­pu­jące pole­ce­nie w ter­mi­nalu, będąc w głów­nym kata­logu pro­jektu:

    python gomoku.py

## Zasady gry

Celem gry jest uło­że­nie pię­ciu pion­ków tego samego koloru w jed­nym, cią­głym rzę­dzie. Może to być linia pozioma, pio­nowa lub dia­go­nalna.

Gra toczy się na plan­szy o wymia­rach **10x10**

Gra­cze na zmianę sta­wiają swoje pionki na plan­szy, poda­jąc współ­rzędne:

    x,y

Pierw­szy gracz, który ułoży **pięć pion­ków w rzę­dzie**, wygrywa.

Jeśli cała plan­sza zosta­nie zapeł­niona i żaden z gra­czy nie ułoży pię­ciu w rzę­dzie, gra koń­czy się remi­sem.

## Zaim­ple­men­to­wane algo­rytmy

### Nega­max: 
Jest to wariant algo­rytmu mini­max prze­zna­czony do gier o sumie zero­wej. Działa na zasa­dzie, że gracz, który ma się ruszyć, chce zmak­sy­ma­li­zo­wać swoją ocenę, pod­czas gdy jego prze­ciw­nik chce ją zmi­ni­ma­li­zo­wać.

### Heu­ry­styka punk­towa: 
W kodzie znaj­duje się funk­cja sco­ring, która oce­nia stan plan­szy, dając wyż­szą war­tość cią­gom pion­ków wła­snego gra­cza i niż­szą cią­gom pion­ków prze­ciw­nika. Zapew­nia to, że SI podej­muje stra­te­giczne decy­zje, dążąc do two­rze­nia wła­snych linii, jed­no­cze­śnie blo­ku­jąc prze­ciw­nika.

************************

## Zrzut ekranu z roz­grywki

************************

## Auto­rzy

*Jan Sku­li­mow­ski (s27144)*

*Kamil Lit­twitz (s26966)*

************************
<p align="center">
<img src="https://github.com/janskulimowski/NAI-GOMOKU/blob/main/gm-image.jpg?raw=true" width="500" alt="Gomoku" />
</p>

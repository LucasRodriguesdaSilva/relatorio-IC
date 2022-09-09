/*
* NOME: LUCAS RODRIGUES DA SILVA 
* MATRÍCULA: 428787
* REDE NEURAL MLP
*/

clear();
clc();

//pegando os dados da base dermatology.
base_original = csvRead('dermatology.data');

/*
* Retirando a linha dos dados faltantes
*/

r = isnan(base_original(:,34));
[t u] = size(base_original);

base = [];
cont = 1
for i=1:t
    if r(i) == %F
        base(cont,:) = base_original(i,:);
        cont = cont + 1;
    end
end

/*
* Extraindo as amostras em vetores de entrada e vetores de rótulos
*/
[l m] = size(base);
// Ordenando as classes em ordem crescente de valores
[s,k] = (gsort(base(:,35),'lr','i'));

base_final = [];
for i=1:l
    base_final(i,:) = base(k(i),:);
end

x = base_final(:,1:34)'; // Vetores de entrada
// Pegando as classes e transformando na notação [1 0 0 0 0 0]
y = zeros(l,6);
for k=1:l
    aux = base_final(k,35);
    y(k,aux) = 1;
end
y = y'; // Vetores de rotulos

// Normalizando os valores de x por z-score
for i=1:34
    x(i,:) = (x(i,:) - mean(x(i,:)))/ stdev(x(i,:));
end

/*
* Separando os valores em treino e teste
*/

//escolhendo os índices das amostras para treino e teste, 50% de cada classe.
index_treino = [1:55,112:142,172:206,243:266,291:314,339:348];  
index_teste =  [56:111,143:171,207:242,267:290,315:338,349:358];

x_treino = x(:,index_treino);
y_treino = y(:,index_treino);

x_teste = x(:,index_teste);
y_teste = y(:,index_teste);

/*
* Treinando e testando a rede
*/

// Treino e teste
function [C]=train_prev(n)
    n_entrada = 34; // número de vetores de entrada
    n_rotulos = 6; // número de vetores de rótulos

    // ANN FeedForward Backpropagation Gradient Decent training function.
    W = ann_FFBP_gd(x_treino,y_treino,[n_entrada n n_rotulos]);

    // Prevendo o W com os dados de x
    C = ann_FFBP_run(x_teste,W); 
endfunction

// Calculando a quantidade de acertos
function [cont]= calcular_acertos(y_teste, C)
    cont=0;
    for i=1:(l/2)
        [a b] = max(y_teste(:,i));
        [c d] = max(C(:,i));
        if b == d
            cont = cont + 1;
        end    
    end
endfunction

// Número de neurônios na camada oculta
n_neuronios = [50 100 175]; 

for i=1:3
    C = train_prev(n_neuronios(i)); // treinamento e previsão 
    cont(i) = calcular_acertos(y_teste,C); // Número de acertos
end

//Acertos da rede MLP.
for i=1:3
    result = cont(i)/(l/2);
    disp('--------- REDE MLP COM ' + string(n_neuronios(i)) + ' NEURÔNIOS OCULTOS -------------'); 
    disp('A percentagem de acertos da rede MLP com ' + string(n_neuronios(i)) + ' neurônios ocultos foi de: ' + string(result));
end



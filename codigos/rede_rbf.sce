/*
* NOME: LUCAS RODRIGUES DA SILVA 
* MATRÍCULA: 428787
* REDE NEURAL RBF
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

// Função que treina a rede
function [W] = treino(x_treino,y_treino, T, q, N, sigma)
    Z_treino = zeros(q,(N/2));
    

    for i=1:(N/2)
        for j=1:q
            v = norm(x_treino(:,i) - T(:,j));
            Z_treino(j,i) = exp(-v^2/2*(sigma^2));
        end
    end

    Z_treino = [ones(1,(N/2));Z_treino];

    W = (y_treino*Z_treino') * ((Z_treino*Z_treino')^(-1));
    
endfunction

// Função que testa a rede
function [cont] = teste(x_teste,y_teste,W,T,q,N, sigma)
    Z_teste = zeros(q,(N/2));

    for i=1:(N/2)
        for j=1:q
            v = norm(x_teste(:,i) - T(:,j));
            Z_teste(j,i) = exp(-v^2/2*(sigma^2));
        end
    end

    Z_teste = [ones(1,(N/2));Z_teste];
    prev = W * Z_teste;

    cont = 0;
    for k=1:(N/2)
        [a b] = max(prev(:,k));
        [c d] = max(y_teste(:,k));
        if b == d
            cont = cont + 1;
        end
    end
endfunction

[p N] = size(x);
sigma = 0.3; 
// p é a quantidade de atributos em cada valor de entrada x
q = [2 10 25]; // Quantidade de neurônios ocultos

// Testando e Treinando a rede com as amostras.
for i=1:3
    T = rand(p,q(i),'normal'); // Vetores centroide dos q neurônios
    W = treino(x_treino,y_treino,T,q(i),N,sigma);
    cont(i) = teste(x_teste,y_teste,W,T,q(i),N, sigma);
end

//Acertos da rede RBF.
for i=1:3
    result = cont(i)/(N/2);
    disp('--------- REDE RBF COM ' + string(q(i)) + ' NEURÔNIOS OCULTOS -------------'); 
    disp('A percentagem da rede RBF com ' + string(q(i)) + ' neurônios ocultos foi de: ' + string(result));
end


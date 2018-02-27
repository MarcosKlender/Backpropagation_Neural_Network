#include <math.h>

class TNeuralNetwork
{
  public:
    TNeuralNetwork(char, TList *, float,float);					//Capas, Neuronas Por Capa, ETA, BIAS
    ~TNeuralNetwork();
    TList* outVector;											//Última salida (vector)
    char train(TList *, TList *);
    char run(TList *);
    float sqError;												//Error Cuadrático de la Red

  private:
    float eta;													//Coeficiente de Aprendizaje
    float bias;													//Umbral
    TList *layers;
    TList *outputs;												//Salidas para el arreglo layers[i]
    TList *neuronsPerLayer;
    int inputCount;
    int outputCount;
    char layerCount;
};

class TNeuron
{
  public:
    TNeuron(int);    											//Entradas
    ~TNeuron();
    float output;												//Salidas
    TList *inputs;												//Lista de Entradas
    float sigmoid(float);
    float net();											    //Sumatoria Ponderada
    float errorTerm;											//Delta
    TList *deltas;												//Cambios en los Pesos
    TList *weights;  											//Pesos como Entradas

  private:
    int inputCount;
};

TNeuron::~TNeuron(){}
TNeuron::TNeuron(int inputsValue)
{
  inputCount=inputsValue;
  deltas = new TList();
  weights = new TList();
  for (int i=0;i<=inputCount;i++)
  {
    deltas->Add(malloc(sizeof(float)));
    weights->Add(malloc(sizeof(float)));
    *((float *)(weights->Items[i]))=(float)(rand()/(32767.0/2)-1);
  }
}

float TNeuron::sigmoid(float num)
{
   return (float)(1/(1+exp(-num)));
}

float TNeuron::net()
{
  float acc=0;
  for (int i=0;i<inputCount;i++)
  {
    acc+=(*((float *)inputs->Items[i])) * (*((float *)weights->Items[i]));
  }
  return acc;
}

TNeuralNetwork::~TNeuralNetwork(){}
TNeuralNetwork::TNeuralNetwork(char layersValue, TList *neuronsPerLayerValue, float etaValue, float biasValue)
{
  randomize();
  bias=biasValue;
  eta=etaValue;
  neuronsPerLayer=new TList();
  neuronsPerLayer->Assign(neuronsPerLayerValue,laCopy,NULL);
  layerCount=layersValue;
  inputCount= *((int *)(neuronsPerLayer->Items[0]));
  outputCount= *((int *)(neuronsPerLayer->Items[layerCount-1]));
  layers = new TList();
  outputs = new TList();
  for (char i=0;i<layerCount;i++)													//Creación de Neuronas
  {
    layers->Add(new TList());
    outputs->Add(new TList());
    ((TList *)layers->Items[i])->Add(NULL);      									//Primera Neurona
    ((TList *)(outputs->Items[i]))->Add(NULL);   									//Primera Salida
    char tope = *((char *)(neuronsPerLayer->Items[i]));
    for (char j=1; j<=tope;j++)														//Interconexión entre Entradas y Salidas
    {
      char temp = (!i)?0:(*((char *)(neuronsPerLayer->Items[i-1])));
      ((TList *)layers->Items[i])->Add(new TNeuron(temp+1));
      TNeuron *out=(TNeuron *)((TList *)((TList *)layers->Items[i])->Items[j]);
      TList *outMat= (TList *)(outputs->Items[i]);
      outMat->Add( &(((TNeuron *)out)->output));
    }
  }

  ((TList *)(outputs->Items[0]))->Items[0] = (float *)(&bias);
  for (char i=1;i<layerCount;i++)													//Interconexión de la Red
  {
    ((TList *)(outputs->Items[i]))->Items[0] = (float *)(&bias);
    char tope = *((char *)(neuronsPerLayer->Items[i]));
    for (char j=1; j<=tope;j++)
    {
      TList *temp=(TList *)(layers->Items[i]);
      TNeuron *neu=(TNeuron *)(temp->Items[j]);
      neu->inputs=(TList *)(outputs->Items[i-1]);
    }
  }
  outVector = (TList *) ((TList *)outputs->Items[layerCount-1]);
}

char TNeuralNetwork::run(TList *pattern)
{
  TList *inputs=((TList *)(layers->Items[0]));
  for (int i=1;i<=inputCount;i++)
  {
    ((TNeuron *)(inputs->Items[i]))->output = *((float*)(TList *)pattern->Items[i-1]);
  }
  for (char i=1;i<layerCount;i++)
  {
    char tope = *((char *)(neuronsPerLayer->Items[i]));
    for (char j=1; j<=tope;j++)
    {
      TNeuron *temp=(TNeuron *)((TList *)layers->Items[i])->Items[j];
      temp->output=temp->sigmoid(temp->net());
    }
  }
  return(1);
}

char TNeuralNetwork::train(TList *pattern, TList *expected)							//BACKPROPAGATION
{
//PASO 1: Iniciar la Red
  run(pattern);

//PASO 2: Calcular el Error Cuadrático de la Red
  sqError=0;
  for(int i=1;i<=outputCount;i++)
  {
    float error, actual, desired;
    actual =  *((float *) ((TList *)(outputs->Items[layerCount-1]))->Items[i]);
    desired = *((float *) (expected->Items[i-1]));
    error = desired-actual;
    error *= error;
    sqError += error;
  }
  sqError=(float)sqError/(outputCount+0.0);

  for(int x=(layerCount-1);x>=1;x--)
  {
//PASO 3: Calcular el Error de cada una de las Capas
    int layersHere = *((int*)(neuronsPerLayer->Items[x]));
    for(int i=1;i<=layersHere;i++)
    {
      float error, actual, desired, *errorTerm;  									 //Valores Deseados
      TNeuron *neuron = ((TNeuron *)((TList *)(layers->Items[x]))->Items[i]);
      errorTerm = &(neuron->errorTerm);
      actual =  *((float *) ((TList *)(outputs->Items[x]))->Items[i]);
      if(x==(layerCount-1))
      {
        desired = *((float *) (expected->Items[i-1]));
        error = desired-actual;
        *errorTerm = actual*(1.0-actual)*(error);
      }
      else
      {
        error=0;
        int layersAfter = *((int*)(neuronsPerLayer->Items[x+1]));
        for(int j=1;j<=layersAfter;j++)
        {
          TNeuron *neuronFwd = ((TNeuron *)((TList *)(layers->Items[x+1]))->Items[j]);
          error+= (*((float *)(neuronFwd->weights->Items[i]))) * neuronFwd->errorTerm;
        }
      }
      *errorTerm = actual*(1.0-actual)*(error);
    }

//PASO 4 & 5: Calcular los Deltas y Actualizar los Pesos
    int layersBefore = *((int*)(neuronsPerLayer->Items[x-1]));
    for(int i=1;i<=layersHere;i++)
    {
      TList *deltas, *inputs, *weights;
      float errorTerm;
      TNeuron *neuron = ((TNeuron *)((TList *)(layers->Items[x]))->Items[i]);
      deltas = neuron->deltas;
      inputs = neuron->inputs;
      weights = neuron->weights;
      errorTerm = neuron->errorTerm;
      for(int j=0;j<=layersBefore;j++)
      {
        *((float *)(deltas->Items[j]))= eta * errorTerm * (*((float *)(inputs->Items[j])));
        *((float *)(weights->Items[j]))+= (*((float *)(deltas->Items[j])));
      }
    }
  }
  return(1);
}
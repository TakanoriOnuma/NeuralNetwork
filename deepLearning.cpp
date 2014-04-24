#include <time.h>

#include <iostream>
#include <vector>

using namespace std;

double fout(double x)
{
    return 1 / (1 + exp(-x));
}

// dec_point_num <= 3‚ª‚¢‚¢B4‚©‚ç­‚µ“®ì‚ª•Ï
double my_rand(double min, double max, int dec_point_num)
{
    double value = 1.0;
    for(int i = 0; i < dec_point_num; i++){
        value *= 10.0;
    }
    return (double)(rand() % (int)((max - min) * value + 1)) / value + min;
}

class Neuron {
    double u;
    double x;
    vector<double> w;
    double bias;

public:
    Neuron(const vector<double>& w, double bias)
        : w(w), bias(bias), u(0.0), x(0.0)
    {
    }

    // --- getter --- //
    double getU() const { return u; }
    double getX() const { return x; }
    const vector<double>& getW() const { return w; }
    double getBias() const { return bias; }

    void scalarProduct(const vector<double>& in) {
        if(w.size() == in.size()) {
            u = 0.0;
            for(int i = 0; i < w.size(); i++) {
                u += w[i] * in[i];
            }
            x = fout(u + bias);
        }
    }
};

void forwardPropagation(vector<vector<Neuron*>>& neurons, const vector<double>& inp)
{
    neurons[0][0]->scalarProduct(inp);
    for(int i = 1; i < neurons.size(); i++) {
        vector<double> inp(neurons[i - 1].size());
        for(int j = 0; j < inp.size(); j++) {
            inp[j] = neurons[i - 1][j]->getX();
        }
        for(int j = 0; j < neurons[i].size(); j++) {
            neurons[i][j]->scalarProduct(inp);
        }
    }
}

int main()
{
    srand((unsigned int)time(NULL));

    vector<vector<Neuron*>> neurons(0, vector<Neuron*>(0));

    neurons.resize(5);
    neurons[0].resize(1);
    neurons[1].resize(5);
    neurons[2].resize(3);
    neurons[3].resize(5);
    neurons[4].resize(1);

    vector<double> w(1);
    w[0] = 1.0;
    neurons[0][0] = new Neuron(w, 0.0);

    for(int i = 1; i < neurons.size(); i++) {
        w = vector<double>(neurons[i - 1].size());
        for(int j = 0; j < neurons[i].size(); j++) {
            for(int k = 0; k < neurons[i - 1].size(); k++) {
                w[k] = my_rand(-1, 1, 2);
            }
            neurons[i][j] = new Neuron(w, my_rand(-1, 1, 2));
        }
    }

    vector<double> inp(1);
    inp[0] = my_rand(-1, 1, 2);
    forwardPropagation(neurons, inp);


    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            cout << "neurons[" << i << "][" << j << "]:"
                << neurons[i][j]->getW().size() << ", " << neurons[i][j]->getBias() << ", "
                << neurons[i][j]->getU() << ", " << neurons[i][j]->getX() << endl;
      
            for each(double w in neurons[i][j]->getW()) {
                cout << w << ", ";
            }
            cout << endl;
        }
    }

    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            delete neurons[i][j];
        }
    }  

    return 0;
}
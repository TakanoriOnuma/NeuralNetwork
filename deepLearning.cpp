#include <time.h>

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;


const int    N        = 16;         // sin�g�̃v���b�g�����
const int    NUM_STEP = 10000;      // 1��̃��[�v�Ŋw�K�������
const int    Patterns = 30;
const double Eta      = 0.03;
const double Alpha    = 0.8;
const double PAI      = 3.14159265359;
const double ErrorEv  = 0.03;
const double Rlow     = -1.0;
const double Rhigh    = 1.0;

inline double fout(double x)
{
    return tanh(x);
}

inline double delta_fout(double fout_value)
{
    return 1 - fout_value * fout_value;
}

// dec_point_num <= 3�������B4���班�����삪��
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
    vector<double> delta_w;
    double bias;
    double delta_bias;

public:
    Neuron(const vector<double>& w, double bias)
        : w(w), bias(bias), u(0.0), x(0.0), delta_w(0), delta_bias(0.0)
    {
        delta_w.resize(w.size());
        for(int i = 0; i < delta_w.size(); i++) {
            delta_w[i] = 0.0;
        }
    }

    // --- getter --- //
    double getU() const { return u; }
    double getX() const { return x; }
    const vector<double>& getW() const { return w; }
    const vector<double>& getDeltaW() const { return delta_w; }
    double getBias() const { return bias; }
    double getDeltaBias() const { return delta_bias; }

    // --- setter --- //
    void setW(const vector<double>& w) { this->w = w; }
    void setDeltaW(const vector<double>& delta_w) { this->delta_w = delta_w; }
    void setBias(double bias) { this->bias = bias; }
    void setDeltaBias(double delta_bias) { this->delta_bias = delta_bias; }

    void scalarProduct(const vector<double>& in) {
        if(w.size() == in.size()) {
            u = 0.0;
            for(int i = 0; i < w.size(); i++) {
                u += w[i] * in[i];
            }
            x = fout(u + bias);
        }
    }

    // ���͑w�̓��͐M���͂��̂܂ܓ����
    void setX(double x) { this->x = x; }
};

void forwardPropagation(vector<vector<Neuron*>>& neurons, const vector<double>& inp)
{
    for(int i = 0; i < neurons[0].size(); i++) {
        neurons[0][i]->setX(inp[i]);
    }
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

void backPropagation(vector<vector<Neuron*>>& neurons, const vector<double>& tsignal)
{

    vector<Neuron*>& out_neurons = neurons[neurons.size() - 1];     // �o�͑w�j���[����
    // ���͐M���Ƌ��t�M���Ƃ̌덷�����߂�
    vector<double> dwo(out_neurons.size());
    for(int i = 0; i < dwo.size(); i++) {
        dwo[i] = (tsignal[i] - out_neurons[i]->getX()) * delta_fout(out_neurons[i]->getX());
    }


    // �o�͑w�̌����׏d�l��ύX����
    for(int i = 0; i < out_neurons.size(); i++) {
        vector<double> w       = out_neurons[i]->getW();
        vector<double> delta_w = out_neurons[i]->getDeltaW();
        for(int j = 0; j < w.size(); j++) {
            delta_w[j] = Eta * dwo[i] * neurons[neurons.size() - 2][j]->getX() + Alpha * delta_w[j];
            w[j] += delta_w[j];
        }
        out_neurons[i]->setW(w);
        out_neurons[i]->setDeltaW(delta_w);
    }

    // �o�͑w�̂������l��ύX����
    for(int i = 0; i < out_neurons.size(); i++) {
        double bias = out_neurons[i]->getBias();
        double delta_bias = out_neurons[i]->getDeltaBias();
        delta_bias = Eta * dwo[i] + Alpha * delta_bias;
        bias += delta_bias;
        out_neurons[i]->setBias(bias);
        out_neurons[i]->setDeltaBias(delta_bias);
    }

    // ���ԑw�̌����׏d�l�Ƃ������l��ύX����
    vector<double> up_delta  = dwo;
    for(int i = (int)neurons.size() - 2; i >= 1; i--) {
        vector<double> now_delta(neurons[i].size());

        // �덷�`���Ō덷�����߂�
        for(int j = 0; j < neurons[i].size(); j++) {
            double sum = 0.0;
            
            for(int k = 0; k < neurons[i + 1].size(); k++) {
                sum += up_delta[k] * neurons[i + 1][k]->getW()[j];
            }
            now_delta[j] = delta_fout(neurons[i][j]->getX()) * sum;
        }

        // ���݂̑w�̌����׏d�l��ύX����
        for(int j = 0; j < neurons[i].size(); j++) {
            vector<double> w = neurons[i][j]->getW();
            vector<double> delta_w = neurons[i][j]->getDeltaW();
            for(int k = 0; k < w.size(); k++) {
                delta_w[k] = Eta * now_delta[j] * neurons[i - 1][k]->getX() + Alpha * delta_w[k];
                w[k] += delta_w[k];
            }
            neurons[i][j]->setW(w);
            neurons[i][j]->setDeltaW(delta_w);
        }

        // ���݂̑w�̂������l��ύX����
        for(int j = 0; j < neurons[i].size(); j++) {
            double bias = neurons[i][j]->getBias();
            double delta_bias = neurons[i][j]->getDeltaBias();
            delta_bias = Eta * now_delta[j] + Alpha * delta_bias;
            bias += delta_bias;
            neurons[i][j]->setBias(bias);
            neurons[i][j]->setDeltaBias(delta_bias);
        }

        up_delta = now_delta;
    }
}

double calcError(vector<vector<Neuron*>>& neurons, const vector<double> inp_dats[], const vector<double> tsignal[], const int patterns)
{
    double error = 0.0;
    const vector<Neuron*>& out_neurons = neurons[neurons.size() - 1];

    // ��A�̊w�K�f�[�^���J��Ԃ��āA�덷���W�v����
    for(int i = 0; i < patterns; i++) {
        forwardPropagation(neurons, inp_dats[i]);

        for(int j = 0; j < out_neurons.size(); j++) {
            error += sqrt(pow(tsignal[i][j] - out_neurons[j]->getX(), 2.0));
        }
    }
    error *= 0.5;

    return error;
}

void outNetworkProperty(const char* filename, const vector<vector<Neuron*>>& neurons)
{
    ofstream ofs(filename);

    ofs << "N:"       << N        << endl;
    ofs << "Eta:"     << Eta      << endl;
    ofs << "Alpha:"   << Alpha    << endl;
    ofs << "ErrorEv:" << ErrorEv  << endl;
    ofs << "Rlow:"    << Rlow     << endl;
    ofs << "Rhigh:"   << Rhigh    << endl;
    ofs << "NumStep:" << NUM_STEP << endl;
    ofs << "--- Neuron �K�w:" << neurons.size() << " ---" << endl;
    for(int i = 0; i < neurons.size(); i++) {
        ofs << "��" << (i + 1) << "�w:" << "�j���[����" << neurons[i].size() << "��" << endl;
    }
}

// ���ԑw�̏o�͌��ʂ��t�@�C���ɏo��
void outMiddleData(const char* filename, vector<vector<Neuron*>>& neurons)
{
    ofstream ofs_middle(filename);  // ���ԑw�̃f�[�^
    ofs_middle << "# A" << "\t" << "��" << "\t";
    for(int j = 0; j < neurons[neurons.size() / 2].size(); j++) {
        ofs_middle << "neuron[" << neurons.size() / 2 << "][" << j << "]" << "\t";
    }
    ofs_middle << endl;

    // ���ԑw�̏o�͂����߂�
    for(double Lambda = PAI; Lambda <= 2 * PAI; Lambda += 0.1) {
        ofs_middle << "# ��=" << Lambda << endl;
        for(double A = 0.1; A <= 0.8; A += 0.02) {
            // ���͂���sin�g�����
            vector<double> sin_dat;
            for(int i = 0; i < N + 1; i++) {
                double in_data = 2.0 * i / N - 1.0;
                sin_dat.push_back(A * sin(Lambda * in_data));
            }
        
            forwardPropagation(neurons, sin_dat);

            // ���ԑw�̏o�͂��t�@�C���ɏo��
            ofs_middle << A << "\t" << Lambda << "\t";
            const vector<Neuron*>& mid_neurons = neurons[neurons.size() / 2];
            for(int i = 0; i < mid_neurons.size(); i++) {
                ofs_middle << mid_neurons[i]->getX() << "\t";
            }
            ofs_middle << endl;
        }
        ofs_middle << endl;
    }
}


// �w�K���sin�f�[�^���t�@�C���ɏo��
void outLearningSinData(const char* filename, vector<vector<Neuron*>> neurons, const vector<double> inp_dats[], const vector<double> tsignal[])
{
    // ���ʂ��o��
    ofstream ofs_sin("out_sin.dat");
    ofs_sin << "# x" << "\t";
    for(int i = 0; i < Patterns; i++) {
        ofs_sin << "tsignal[" << i << "]\t" << "output[" << i << "]\t";
    }
    ofs_sin << endl;

    double out_sin[Patterns][N + 1];        // sin�̊w�K����

    // �w�K���sin�����߂�
    for(int i = 0; i < Patterns; i++) {
        forwardPropagation(neurons, inp_dats[i]);
        for(int j = 0; j < neurons[neurons.size() - 1].size(); j++) {
            out_sin[i][j] = neurons[neurons.size() - 1][j]->getX();
        }
    }

    for(int i = 0; i < N + 1; i++) {
        ofs_sin << (2.0 * i / N - 1.0) << "\t";
        for(int j = 0; j < Patterns; j++) {
            ofs_sin << tsignal[j][i] << "\t" << out_sin[j][i] << "\t";
        }
        ofs_sin << endl;
    }

}

int main()
{
    srand((unsigned int)time(NULL));

    vector<vector<Neuron*>> neurons(0, vector<Neuron*>(0));

    // �j���[������5�w��N+1-7-3-7-N+1�ɂ���
    neurons.resize(5);
    neurons[0].resize(N + 1);
    neurons[1].resize(7);
    neurons[2].resize(3);
    neurons[3].resize(7);
    neurons[4].resize(N + 1);

    vector<double> w(1);
    w[0] = 1.0;
    for(int i = 0; i < N + 1; i++) {
        neurons[0][i] = new Neuron(w, 0.0);
    }
    // �j���[�����l�b�g���[�N���\�z����
    for(int i = 1; i < neurons.size(); i++) {
        w = vector<double>(neurons[i - 1].size());
        for(int j = 0; j < neurons[i].size(); j++) {
            for(int k = 0; k < neurons[i - 1].size(); k++) {
                w[k] = my_rand(Rlow, Rhigh, 2);
            }
            neurons[i][j] = new Neuron(w, my_rand(Rlow, Rhigh, 2));
        }
    }

    outNetworkProperty("neuron_property.txt", neurons);


    // ���t�f�[�^�̍쐬
    vector<double> inp_dats[Patterns];
    vector<double> tsignal[Patterns];

    ofstream ofs_tsignal("tsignal.dat");
    ofs_tsignal << "# pattern\t" << "A\t" << "Lambda\t" << endl;
    for(int i = 0; i < Patterns; i++) {
        double A = my_rand(0.1, 0.8, 2);
        double Lambda = my_rand(PAI, 2 * PAI, 2);
        for(int j = 0; j < N + 1; j++) {
            double inp_data = 2.0 * j / N - 1.0;
            double sin_data = A * sin(Lambda * inp_data);
            inp_dats[i].push_back(sin_data);
            tsignal[i].push_back(sin_data);
        }
        ofs_tsignal << i << "\t" << A << "\t" << Lambda << endl;
    }
    
    ofstream ofs_err("error.dat");
    ofs_err << "# " << "step\t" << "error" << endl;
    ofstream ofs_w("out_w.dat");
    ofs_w << "# " << "step\t";
    for(int i = 1; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            const vector<double>& w = neurons[i][j]->getW();
            for(int k = 0; k < w.size(); k++) {
                ofs_w << "neurons[" << i << "][" << j << "]->w[" << k << "]\t";
            }
        }
    }
    ofs_w << endl;
    // �w�K������
    double vError = calcError(neurons, inp_dats, tsignal, Patterns);
    for(int i = 0; vError > ErrorEv && i < 10; i++) {
        // �t�@�C���ɏo��
        ofs_err << i << "\t" << vError << endl;
        cout << vError << endl;
        ofs_w << i << "\t";
        for(int ii = 1; ii < neurons.size(); ii++) {
            for(int j = 0; j < neurons[ii].size(); j++) {
                const vector<double>& w = neurons[ii][j]->getW();
                for(int k = 0; k < w.size(); k++) {
                    ofs_w << w[k] << "\t";
                }
            }
        }
        ofs_w << endl;

        for(int step = 0; step < NUM_STEP; step++) {
            for(int j = 0; j < Patterns; j++) {
                forwardPropagation(neurons, inp_dats[j]);
                backPropagation(neurons, tsignal[j]);
            }
        }

        vError = calcError(neurons, inp_dats, tsignal, Patterns);
    }

    // �j���[�����f�[�^�̏o��
    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            cout << "neurons[" << i << "][" << j << "]:"
                << neurons[i][j]->getW().size() << ", " << neurons[i][j]->getBias() << ", "
                << neurons[i][j]->getU() << ", " << neurons[i][j]->getX() << endl;
      
            for each(double w in neurons[i][j]->getW()) {
                cout << w << ", ";
            }
            cout << endl;
            for each(double delta_w in neurons[i][j]->getDeltaW()) {
                cout << delta_w << ", ";
            }
            cout << endl;
        }
    }

    outMiddleData("out_middle.dat", neurons);
    outLearningSinData("out_sin.dat", neurons, inp_dats, tsignal);

    // �j���[�����̍폜�i���I�Ɋm�ۂ������߁j
    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            delete neurons[i][j];
        }
    }  

    return 0;
}
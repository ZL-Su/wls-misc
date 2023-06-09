#include <algorithm>

// C++11, 14, 17, 20, 23
// Compilation (complile-time)->linker->(runtime)

constexpr auto x = 1024 / 2;

namespace dyvp
{
   template <typename T, size_t N>
   class Vector
   {
      using _Myt = Vector;

   public:
      using value_type = std::decay_t<T>;
      inline Vector(size_t _ = 1) noexcept
          : dim(_)
      {
      }

      inline decltype(auto) operator[](size_t i) const
      {
         return (pd[i]);
      }
      inline decltype(auto) operator[](size_t i)
      {
         return pd[i];
      }
      inline auto size() const noexcept
      {
         return dim;
      }

   protected:
      value_type data[N];
      size_t dim{N};
      value_type *pd = data;
   };

   template <typename T>
   class Vector<T, 0> : Vector<T, 1>
   {
      using _Mybase = Vector<T, 1>;

   public:
      using typename _Mybase::value_type;
      using _Mybase::operator[];
      using _Mybase::size;

      Vector(size_t N, value_type v = 0)
          : _Mybase(N)
      {
         this->pd = new value_type[_Mybase::dim];
         std::fill(this->pd, this->pd + size(), v);
         this->data[0] = this->pd[0];
      }
      ~Vector()
      {
         delete[] this->pd;
         this->pd = this->data;
      }
   };

   template <typename T, size_t N>
   inline Vector<T, N> sum(const Vector<T, N> &x, const Vector<T, N> &y) noexcept
   {
      Vector<T, N> ret(std::min(x.size(), y.size()));
      for (auto i = 0; i < ret.size(); ++i)
      {
         ret[i] = x[i] + y[i];
      }
      return ret;
   }
   template <typename T, size_t N>
   inline Vector<T, N> operator-(const Vector<T, N> &x, const Vector<T, N> &y) noexcept
   {
      Vector<T, N> ret(std::min(x.size(), y.size()));
      for (auto i = 0; i < ret.size(); ++i)
      {
         ret[i] = x[i] - y[i];
      }
      return ret;
   }
   template <typename T, size_t N>
   inline Vector<T, N> operator*(const Vector<T, N> &x, const Vector<T, N> &y) noexcept
   {
      Vector<T, N> ret(std::min(x.size(), y.size()));
      for (auto i = 0; i < ret.size(); ++i)
      {
         ret[i] = x[i] * y[i];
      }
      return ret;
   }
   template <typename T, size_t N>
   inline Vector<T, N> operator/(const Vector<T, N> &x, const Vector<T, N> &y) noexcept
   {
      Vector<T, N> ret(std::min(x.size(), y.size()));
      for (auto i = 0; i < ret.size(); ++i)
      {
         ret[i] = x[i] / y[i];
      }
      return ret;
   }
}
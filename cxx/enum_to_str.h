#include <string>

/*
 * The following code snappt
 */

#define DOGS                    \
   {                            \
      C(JACK), C(BULL), C(ITAL) \
   }
#undef C
#define C(a) ENUM_##a
enum dogs DOGS;
#undef C
#define C(a) #a

std::string dog_strings[] = DOGS;
auto dog_to_string(enum dogs name)
{
   return dog_strings[name];
}